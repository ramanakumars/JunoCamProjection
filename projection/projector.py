import numpy as np
import json
import re
from skimage import feature
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import os
import time
import matplotlib.pyplot as plt
import multiprocessing
import spiceypy as spice
from skimage import io
import tqdm
from ftplib import FTP
from .globals import FRAME_HEIGHT, FRAME_WIDTH, initializer
from .cython_utils import furnish_c, project_midplane_c, get_pixel_from_coords_c
from .camera_funcs import CameraModel
import sys
import healpy as hp

# for decompanding -- taken from Kevin Gill's github page
SQROOT = np.array(
    (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
        47, 49, 51, 53, 55, 57, 59, 61, 63, 67, 71, 75, 79, 83, 87, 91,
        95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147,
        151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203,
        207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 255, 263, 271,
        279, 287, 295, 303, 311, 319, 327, 335, 343, 351, 359, 367, 375, 383,
        391, 399, 407, 415, 423, 431, 439, 447, 455, 463, 471, 479, 487, 495,
        503, 511, 519, 527, 535, 543, 551, 559, 567, 575, 583, 591, 599, 607,
        615, 623, 631, 639, 647, 655, 663, 671, 679, 687, 695, 703, 711, 719,
        727, 735, 743, 751, 759, 767, 775, 783, 791, 799, 807, 815, 823, 831,
        839, 847, 855, 863, 871, 879, 887, 895, 903, 911, 919, 927, 935, 943,
        951, 959, 967, 975, 983, 991, 999, 1007, 1023, 1039, 1055, 1071,
        1087, 1103, 1119, 1135, 1151, 1167, 1183, 1199, 1215, 1231, 1247,
        1263, 1279, 1295, 1311, 1327, 1343, 1359, 1375, 1391, 1407, 1439,
        1471, 1503, 1535, 1567, 1599, 1631, 1663, 1695, 1727, 1759, 1791,
        1823, 1855, 1887, 1919, 1951, 1983, 2015, 2047, 2079, 2111, 2143,
        2175, 2207, 2239, 2271, 2303, 2335, 2367, 2399, 2431, 2463, 2495,
        2527, 2559, 2591, 2623, 2655, 2687, 2719, 2751, 2783, 2815, 2847,
        2879),
    dtype=np.double
)


def decompand(image):
    """
    Decompands the image from the 8-bit in the public release
    to the original 12-bit shot by JunoCam

    Parameters
    ----------
    image : numpy.ndarray
        8-bit input image

    Outputs
    -------
    data : numpy.ndarray
        Original 12-bit image
    """
    data = np.array(255 * image, dtype=np.uint8)
    ny, nx = data.shape

    def get_sqrt(x):
        return SQROOT[x]
    v_get_sqrt = np.vectorize(get_sqrt)

    data2 = v_get_sqrt(data)

    return data2


class Projector:
    """
    The main projector class that determines the surface intercept points
    of each pixel in a JunoCam image

    Methods
    -------
    load_kernels: Determines and loads the required SPICE kernels
        for processing
    process_n_c : Projects an individual framelet in the JunoCam raw image
    process     : Driver for the projection that handles
        parallel processing
    """

    def __init__(self, imagefolder, meta, kerneldir="."):
        with open(meta, "r") as metafile:
            self.metadata = json.load(metafile)

        self.start_utc = self.metadata["START_TIME"]
        self.fname = self.metadata["FILE_NAME"].replace("-raw.png", "")
        intframe_delay = self.metadata["INTERFRAME_DELAY"].split(" ")

        self.fullimg = plt.imread(imagefolder + "%s-raw.png" % self.fname)

        # ignore color channels in the image
        if len(self.fullimg.shape) == 3:
            self.fullimg = self.fullimg.mean(axis=-1)

        self.sclat = float(self.metadata["SUB_SPACECRAFT_LATITUDE"])
        self.sclon = float(self.metadata["SUB_SPACECRAFT_LONGITUDE"])

        # get the exposure and convert to seconds
        self.exposure = (
            float(
                self.metadata["EXPOSURE_DURATION"].replace("<ms>", "").strip()
            )
            / 1.0e3
        )

        self.frame_delay = float(intframe_delay[0])

        # number of strips
        self.nframelets = int(self.metadata["LINES"] / FRAME_HEIGHT)
        # number of RGB frames
        self.nframes = int(self.nframelets / 3)

        self.download_kernels(kerneldir)

        self.re, _, self.rp = spice.bodvar(spice.bodn2c("JUPITER"), "RADII", 3)
        self.flattening = (self.re - self.rp) / self.re

        # calculate the start time
        self.start_et = spice.str2et(self.start_utc)

        self.savefolder = "%s_proj/" % self.fname

        self.find_jitter(jitter_max=120)

        self.et = np.zeros((self.nframes, 3))

        for n in range(self.nframes):
            for c in range(3):
                ci = CameraModel(c)
                self.et[n, c] = self.start_et + ci.time_bias + self.jitter +\
                    (self.frame_delay + ci.iframe_delay) * n

    def download_kernels(self, KERNEL_DATAFOLDER):
        if KERNEL_DATAFOLDER[-1] != '/':
            KERNEL_DATAFOLDER += '/'

        if not os.path.exists(KERNEL_DATAFOLDER):
            os.mkdir(KERNEL_DATAFOLDER)

        for folder in ['ik', 'ck', 'spk', 'pck', 'fk', 'lsk', 'sclk']:
            if not os.path.exists(KERNEL_DATAFOLDER + folder):
                os.mkdir(KERNEL_DATAFOLDER + folder)

        ftp = FTP('naif.jpl.nasa.gov')
        ftp.login()  # login anonymously
        ftp.cwd('pub/naif/JUNO/kernels/')

        iks = sorted(ftp.nlst("ik/juno_junocam_v*.ti"))
        cks = sorted(ftp.nlst("ck/juno_sc_*.bc"))
        spks1 = sorted(ftp.nlst("spk/spk_*.bsp"))
        spks2 = sorted(ftp.nlst("spk/jup*.bsp"))
        spks3 = sorted(ftp.nlst("spk/de*.bsp"))
        spks4 = sorted(ftp.nlst("spk/juno_struct*.bsp"))
        pcks = sorted(ftp.nlst("pck/pck*.tpc"))
        fks = sorted(ftp.nlst("fk/juno_v*.tf"))
        sclks = sorted(ftp.nlst("sclk/JNO_SCLKSCET.*.tsc"))
        lsks = sorted(ftp.nlst("lsk/naif*.tls"))

        year, month, day = self.start_utc.split("-")
        yy = year[2:]
        mm = month
        dd = day[:2]

        intdate = int("%s%s%s" % (yy, mm, dd))

        kernels = []

        # find the ck and spk kernels for the given date
        ckpattern = r"juno_sc_rec_([0-9]{6})_([0-9]{6})\S*"
        nck = 0
        for ck in cks:
            fname = os.path.basename(ck)
            groups = re.findall(ckpattern, fname)
            if len(groups) == 0:
                continue
            datestart, dateend = groups[0]

            if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                kernels.append(ck)
                nck += 1

        """ use the predicted kernels if there are no rec """
        if nck == 0:
            print("Using predicted CK")
            ckpattern = r"juno_sc_pre_([0-9]{6})_([0-9]{6})\S*"
            for ck in cks:
                fname = os.path.basename(ck)
                groups = re.findall(ckpattern, fname)
                if len(groups) == 0:
                    continue
                datestart, dateend = groups[0]

                if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                    kernels.append(ck)
                    nck += 1

        spkpattern = r"spk_rec_([0-9]{6})_([0-9]{6})\S*"
        nspk = 0
        for spk in spks1:
            fname = os.path.basename(spk)
            groups = re.findall(spkpattern, fname)
            if len(groups) == 0:
                continue
            datestart, dateend = groups[0]

            if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                kernels.append(spk)
                nspk += 1

        """ use the predicted kernels if there are no rec """
        if nspk == 0:
            print("Using predicted SPK")
            spkpattern = r"spk_pre_([0-9]{6})_([0-9]{6})\S*"
            for spk in spks1:
                fname = os.path.basename(spk)
                groups = re.findall(spkpattern, fname)
                if len(groups) == 0:
                    continue
                datestart, dateend = groups[0]

                if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                    kernels.append(spk)
                    nspk += 1

        # if(nck*nspk == 0):
        #    print("ERROR: Kernels not found for the date range!")
        assert nck * nspk > 0, "Kernels not found for the given date range!"

        # load the latest updates for these
        kernels.append(iks[-1])
        kernels.append(spks2[-1])
        kernels.append(spks3[-1])
        kernels.append(spks4[-1])
        kernels.append(pcks[-1])
        kernels.append(fks[-1])
        kernels.append(sclks[-1])
        kernels.append(lsks[-1])
        kernels.append("spk/juno_rec_orbit.bsp")
        kernels.append("spk/juno_pred_orbit.bsp")

        self.kernels = []
        for kernel in kernels:
            kernel_local = KERNEL_DATAFOLDER + kernel
            filesize = ftp.size(kernel)
            try:
                if os.path.isfile(kernel_local) & (os.path.getsize(kernel_local) == filesize):
                    continue
            except FileNotFoundError:
                pass
            print(f"Downloading {kernel}", flush=True)

            with open(kernel_local, 'wb') as kerfile:
                ftp.retrbinary(f"RETR {kernel}", kerfile.write)

        for kernel in kernels:
            kernel_local = KERNEL_DATAFOLDER + kernel
            furnish_c(kernel_local.encode("ascii"))
            spice.furnsh(kernel_local)

            self.kernels.append(kernel_local)

    def find_jitter(self, jitter_max=25, plot=False):
        threshold = 0.1 * self.fullimg.max()

        for nci in range(self.nframes * 3):
            # find whether the planet limb is in now
            # approximating this as the first time the planet is seen
            # in the image, which is generally true..

            n = nci // 3
            ci = nci % 3

            start = 3 * FRAME_HEIGHT * n + ci * FRAME_HEIGHT
            end = 3 * FRAME_HEIGHT * n + (ci + 1) * FRAME_HEIGHT
            frame = self.fullimg[start:end, :]
            if len(frame[frame > threshold]) > 5000:
                # make sure that the limb is also in the image
                # we want to find a frame where both the predicted limb
                # with no jitter and the actual limb are visible so that
                # the offset we determine is small
                cami = CameraModel(ci)
                eti = (
                    self.start_et
                    + cami.time_bias
                    + (self.frame_delay + cami.iframe_delay) * n
                )
                limb_pts = self.get_limb(eti, cami)

                # bad test image if the limb is not fully visible
                # in the frame
                if len(limb_pts) > 5:
                    break

        # create the mask of the visible jupiter in the image
        imgmask = np.zeros_like(frame)
        imgmask[frame > threshold] = 1

        if plot:
            plt.figure(dpi=200)
            plt.imshow(frame, cmap="gray")
            plt.plot(limb_pts[:, 0], limb_pts[:, 1], "r-")
            plt.draw()
            plt.pause(0.001)

        # find the edges
        sigma = 8
        npoints = 0
        while npoints < 10:
            limb_img = np.asarray(
                feature.canny(frame[:, 24:1630], sigma=sigma), dtype=float
            )
            limb_img_points = np.dstack(np.where(limb_img == 1)[::-1])[0]
            limb_img_points[:, 0] += 24
            npoints = len(limb_img_points)
            sigma -= 1

        distances = pairwise_distances(limb_pts, limb_img_points)

        if plot:
            plt.plot(limb_img_points[:, 0], limb_img_points[:, 1], "g-")

        # we've found our limb. now we need to find
        # the actual limb from SPICE
        cami = CameraModel(ci)

        jitter_list = np.arange(-jitter_max, jitter_max, 1) / 1.0e3

        min_dists = np.zeros_like(jitter_list)

        for j, jitter in enumerate(tqdm.tqdm(jitter_list, desc="Finding jitter")):
            eti = self.start_et + cami.time_bias + jitter + (self.frame_delay + cami.iframe_delay) * n

            limbs_jcam = self.get_limb(eti, cami)

            if len(limbs_jcam) < 1:
                min_dists[j] = 1.0e10
                continue

            distances = pairwise_distances(limbs_jcam, limb_img_points)

            if plot:
                plt.figure(dpi=200)
                plt.imshow(frame, cmap="gray")
                plt.plot(
                    limbs_jcam[:, 0], limbs_jcam[:, 1], "r.", markersize=0.1
                )
                plt.plot(
                    limb_img_points[:, 0],
                    limb_img_points[:, 1],
                    "g.",
                    markersize=0.1,
                )
                plt.show()

            if len(distances[distances < 15] > 5):
                min_dists[j] = distances[distances < 15.0].mean()
            else:
                min_dists[j] = 1.0e10

        self.jitter = jitter_list[min_dists.argmin()]

        print("Found best jitter value of %.1f ms" % (self.jitter * 1.0e3))

        eti = (
            self.start_et
            + cami.time_bias
            + self.jitter
            + (self.frame_delay + cami.iframe_delay) * n
        )

        limbs_jcam = self.get_limb(eti, cami)

        if plot:
            plt.plot(limbs_jcam[:, 0], limbs_jcam[:, 1], "g.", markersize=0.1)
            plt.show()

    def get_limb(self, eti, cami):
        METHOD = "TANGENT/ELLIPSOID"
        CORLOC = "ELLIPSOID LIMB"

        scloc, lt = spice.spkpos("JUNO", eti, "IAU_JUPITER", "CN+S", "JUPITER")

        # use the sub spacecraft point as the reference vector
        refvec, _, _ = spice.subpnt("INTERCEPT/ELLIPSOID", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO")

        _, _, eplimbs, limbs_IAU = spice.limbpt(METHOD, "JUPITER", eti, "IAU_JUPITER", "CN+S",
                                                CORLOC, "JUNO", refvec, np.radians(0.1), 3600,
                                                4.0, 0.001, 7200)

        limbs_jcam = np.zeros((limbs_IAU.shape[0], 2))
        for i, limbi in enumerate(limbs_IAU):
            transform = spice.pxfrm2(
                "IAU_JUPITER", "JUNO_JUNOCAM", eplimbs[i], eti
            )
            veci = np.matmul(transform, limbs_IAU[i, :])
            limbs_jcam[i, :] = cami.vec2pix(veci)

        mask = (
            (limbs_jcam[:, 1] < 128)
            & (limbs_jcam[:, 1] > 0)
            & (limbs_jcam[:, 0] > 0)
            & (limbs_jcam[:, 0] < 1648)
        )
        limbs_jcam = limbs_jcam[mask, :]

        return limbs_jcam

    def process(self, nside=2048, num_procs=8, apply_LS=True, n_neighbor=5):
        print("%s" % self.fname)

        lons, lats, incidence, emission, fluxcal, coords, imgvals = self.project_to_midplane(num_procs)

        if apply_LS:
            imgvals = apply_lommel_seeliger(imgvals, incidence, emission)

        coords_new = np.transpose(coords, (1, 0, 2, 3, 4)).reshape(3, -1, 2)
        imgvals_new = np.transpose(imgvals, (1, 0, 2, 3)).reshape(3, -1)

        map = self.project_to_healpix(nside, coords_new, imgvals_new, n_neighbor=n_neighbor)

        return map

    def project_to_midplane(self, num_procs=8):
        """
        Main driver for the projection to a camera frame. Projects
        a full frame onto a camera view of the planet

        Parameters
        ----------
        num_procs : int
            Number of CPUs to use for multiprocessing [Default: 1]
        """

        decompimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        rawimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        # flatfield and gain from Brian Swift's GitHub
        # (https://github.com/BrianSwift/JunoCam/tree/master/Juno3D)
        flatfield = np.array(
            io.imread(
                os.path.dirname(__file__) + "/cal/flatFieldsSmooth12to16.tiff"
            )
        )
        flatfield[flatfield == 0] = 1.0

        inpargs = []
        self.image = np.zeros_like(self.fullimg)

        # decompand the image, apply the flat  field and get the input arguments
        # for the multiprocessing driver
        for n in tqdm.tqdm(range(self.nframes), desc='Decompanding'):
            for c in range(3):
                startrow = 3 * FRAME_HEIGHT * n + c * FRAME_HEIGHT
                endrow = 3 * FRAME_HEIGHT * n + (c + 1) * FRAME_HEIGHT

                framei = decompand(self.fullimg[startrow:endrow, :]) / 16384

                flati = flatfield[
                    (c * FRAME_HEIGHT): ((c + 1) * FRAME_HEIGHT), :
                ]
                framei = framei / flati
                self.image[startrow:endrow, :] = framei / self.exposure
                inpargs.append([self.et[n, c], n, c, np.mean(self.et)])

        with multiprocessing.Pool(
            processes=num_procs, initializer=initializer
        ) as pool:
            try:
                r = pool.map_async(self._project_to_midplane, inpargs)
                pool.close()

                tasks = pool._cache[r._job]
                ninpt = len(inpargs)
                # run the projection using the multiprocessing driver
                with tqdm.tqdm(total=ninpt, desc='Projecting framelets') as pbar:
                    while tasks._number_left > 0:
                        pbar.n = np.max([0, ninpt - tasks._number_left * tasks._chunksize])
                        pbar.refresh()

                        time.sleep(0.1)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                sys.exit()

            pool.join()

        coords = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH, 2))
        imgvals = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        lats = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        lons = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        emission = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        incidence = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        fluxcal = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        results = r.get()

        # fetch the coordinates and the image values
        for jj in range(len(inpargs)):
            loni, lati, inci, emisi, coordsi, fluxcali = results[jj]
            _, i, ci, _ = inpargs[jj]
            startrow = 3 * FRAME_HEIGHT * i + ci * FRAME_HEIGHT
            endrow = 3 * FRAME_HEIGHT * i + (ci + 1) * FRAME_HEIGHT

            # we store both the decompanded and raw images for future use
            decompimg[i, ci, :, :] = self.image[startrow:endrow, :]
            rawimg[i, ci, :, :] = self.fullimg[startrow:endrow, :]

            coords[i, ci, :] = coordsi
            imgvals[i, ci, :] = decompimg[i, ci, :, :] / fluxcali
            lats[i, ci, :] = lati
            lons[i, ci, :] = loni

            # emission and incidence angles for lightning correction
            emission[i, ci, :] = emisi
            incidence[i, ci, :] = inci

            # geometry correction
            fluxcal[i, ci, :] = fluxcali

        return lons, lats, incidence, emission, fluxcal, coords, imgvals

    def _project_to_midplane(self, inpargs):
        eti, n, c, tmid = inpargs

        coords = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2))
        lat = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        lon = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        incidence = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        emission = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        fluxcal = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))

        project_midplane_c(eti, c, tmid, lon, lat, incidence, emission, coords, fluxcal)

        return lon, lat, incidence, emission, coords, fluxcal

    def project_to_healpix(self, nside, coords, imgvals, n_neighbor=4):
        # get the image extents in pixel coordinate space
        x0 = np.nanmin(coords[:, :, 0])
        x1 = np.nanmax(coords[:, :, 0])
        y0 = np.nanmin(coords[:, :, 1])
        y1 = np.nanmax(coords[:, :, 1])

        extents = np.array([x0, x1, y0, y1])

        # now we construct the map grid
        longrid, latgrid = hp.pix2ang(nside, list(range(hp.nside2npix(nside))), lonlat=True)

        pix = np.nan * np.zeros((longrid.size, 2))

        # this assumes the midplane mode where the coordinates
        # are using the Green band camera
        et = np.mean(self.et)

        # get the spacecraft location and transformation to and from the JUNOCAM coordinates
        get_pixel_from_coords_c(np.radians(longrid), np.radians(latgrid), longrid.size, et, extents, pix)

        # get the locations that JunoCam observed
        inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
        pix = pix[inds]
        pixel_inds = hp.ang2pix(nside, longrid[inds], latgrid[inds], lonlat=True)

        # finally, project the image onto the healpix grid
        m = create_image_from_grid(coords, imgvals, pix, pixel_inds, longrid.shape, n_neighbor=n_neighbor)

        return m


def apply_lommel_seeliger(imgvals, incidence, emission):
    '''
        Apply the Lommel-Seeliger correction for incidence
    '''
    # apply Lommel-Siegler correction
    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    corr = 1. / (mu + mu0)
    corr[corr < 1.e-1] = np.nan
    imgvals = imgvals * corr

    return imgvals


def create_image_from_grid(coords, imgvals, pix, inds, img_shape, n_neighbor=5, min_dist=10.):
    '''
        Reproject an irregular spaced image onto a regular grid from a list of coordinate
        locations and corresponding image values. This uses an inverse lookup-table defined
        by `pix`, where pix gives the coordinates in the original image where the corresponding
        pixel coordinate on the new image should be. The coordinate on the new image is given by
        the inds variable.
    '''
    # break up the image and coordinate data into the different filters
    Rcoords = coords[2, :]
    Gcoords = coords[1, :]
    Bcoords = coords[0, :]

    Rmask = np.isfinite(Rcoords[:, 0] * Rcoords[:, 1])
    Gmask = np.isfinite(Gcoords[:, 0] * Gcoords[:, 1])
    Bmask = np.isfinite(Bcoords[:, 0] * Bcoords[:, 1])

    Rcoords = Rcoords[Rmask]
    Gcoords = Gcoords[Gmask]
    Bcoords = Bcoords[Bmask]

    Rimg = imgvals[2, Rmask]
    Gimg = imgvals[1, Gmask]
    Bimg = imgvals[0, Bmask]

    # fit the nearest neighbour algorithm
    print("Fitting neighbors")
    Rneigh = NearestNeighbors().fit(Rcoords)
    Gneigh = NearestNeighbors().fit(Gcoords)
    Bneigh = NearestNeighbors().fit(Bcoords)

    # get the corresponding pixel coordinates in the original image for each filter
    print("Finding neighbors for new image")
    R_dist, R_ind = Rneigh.kneighbors(pix, n_neighbor)
    G_dist, G_ind = Gneigh.kneighbors(pix, n_neighbor)
    B_dist, B_ind = Bneigh.kneighbors(pix, n_neighbor)

    # convert the distance to weights. the farther the point from the
    # target pixel, the less weight it has in the final average
    # add a small epsilon if we accidently are on the exact same point
    R_wght = 1. / (R_dist + 1.e-16)
    G_wght = 1. / (G_dist + 1.e-16)
    B_wght = 1. / (B_dist + 1.e-16)

    # normalize the weights
    R_wght = R_wght / np.sum(R_wght, axis=1, keepdims=True)
    G_wght = G_wght / np.sum(G_wght, axis=1, keepdims=True)
    B_wght = B_wght / np.sum(B_wght, axis=1, keepdims=True)

    R_wght[R_dist.min(axis=1) > min_dist] = 0.
    G_wght[G_dist.min(axis=1) > min_dist] = 0.
    B_wght[B_dist.min(axis=1) > min_dist] = 0.

    # get the weighted NN average for each pixel
    print("Calculating image values at new locations")
    R_vals = np.sum(np.take(Rimg, R_ind, axis=0) * R_wght, axis=1)
    G_vals = np.sum(np.take(Gimg, G_ind, axis=0) * G_wght, axis=1)
    B_vals = np.sum(np.take(Bimg, B_ind, axis=0) * B_wght, axis=1)

    IMG = np.zeros((*img_shape, 3))
    # loop through each point observed by JunoCam and assign the pixel value
    for k, ind in enumerate(tqdm.tqdm(inds, desc='Building image')):
        if len(img_shape) == 2:
            j, i = np.unravel_index(ind, img_shape)

            # do the weighted average for each filter
            IMG[j, i, 0] = R_vals[k]
            IMG[j, i, 1] = G_vals[k]
            IMG[j, i, 2] = B_vals[k]
        else:
            IMG[ind, 0] = R_vals[k]
            IMG[ind, 1] = G_vals[k]
            IMG[ind, 2] = B_vals[k]

    IMG[~np.isfinite(IMG)] = 0.

    return IMG
