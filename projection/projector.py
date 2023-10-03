import numpy as np
import json
import re
from skimage import feature
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt
import spiceypy as spice
from skimage import io
import tqdm
import multiprocessing
import time
import sys
from .globals import FRAME_HEIGHT, FRAME_WIDTH, initializer
from .cython_utils import furnish_c, project_midplane_c, get_pixel_from_coords_c
from .camera_funcs import CameraModel
from .spice_utils import get_kernels
from .frameletdata import FrameletData
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

        print(f"Loading data for {self.fname}")

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

        self.load_kernels(kerneldir)

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

    def load_kernels(self, KERNEL_DATAFOLDER):
        self.kernels = []
        kernels = get_kernels(KERNEL_DATAFOLDER, self.start_utc)
        for kernel in kernels:
            furnish_c(kernel.encode("ascii"))
            spice.furnsh(kernel)
            self.kernels.append(kernel)

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

    def process(self, nside=512, num_procs=8, apply_LS=True, n_neighbor=5):
        print(f"Projecting {self.fname} to a HEALPix grid with n_side={nside}")

        self.project_to_midplane(num_procs)

        if apply_LS:
            self.framedata.update_image(
                apply_lommel_seeliger(self.framedata.image, self.framedata.incidence, self.framedata.emission)
            )

        coords_new = np.transpose(self.framedata.coords, (1, 0, 2, 3, 4)).reshape(3, -1, 2)
        imgvals_new = np.transpose(self.framedata.image, (1, 0, 2, 3)).reshape(3, -1)

        map = self.project_to_healpix(nside, coords_new, imgvals_new, n_neighbor=n_neighbor)

        return map

    def project_to_midplane(self, num_procs=8):
        """
        Main driver for the projection to a camera frame. Projects
        a full frame onto a camera view of the planet at the middle
        timestamp of a given image

        Parameters
        ----------
        num_procs : int
            Number of CPUs to use for multiprocessing [Default: 1]
        """

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

        results = r.get()

        # create the data structure to hold the image and backplane info
        self.framedata = FrameletData(self.nframes)

        # fetch the coordinates and the image values
        for jj in range(len(inpargs)):
            loni, lati, inci, emisi, coordsi, fluxcali = results[jj]
            # loni, lati, inci, emisi, coordsi, fluxcali = _project_to_midplane(*inpargs[jj])
            _, i, ci, _ = inpargs[jj]
            startrow = 3 * FRAME_HEIGHT * i + ci * FRAME_HEIGHT
            endrow = 3 * FRAME_HEIGHT * i + (ci + 1) * FRAME_HEIGHT

            # we store both the decompanded and raw images for future use
            self.framedata.rawimg[i, ci, :, :] = self.image[startrow:endrow, :]

            self.framedata.coords[i, ci, :] = coordsi
            self.framedata.image[i, ci, :] = self.framedata.rawimg[i, ci, :, :] / fluxcali
            self.framedata.lat[i, ci, :] = lati
            self.framedata.lon[i, ci, :] = loni

            # emission and incidence angles for lightning correction
            self.framedata.emission[i, ci, :] = emisi
            self.framedata.incidence[i, ci, :] = inci

            # geometry correction
            self.framedata.fluxcal[i, ci, :] = fluxcali

        return self.framedata

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
        # clip half a pixel to avoid edge artifacts
        x0 = np.nanmin(coords[:, :, 0]) + 0.5
        x1 = np.nanmax(coords[:, :, 0]) - 0.5
        y0 = np.nanmin(coords[:, :, 1]) + 0.5
        y1 = np.nanmax(coords[:, :, 1]) - 0.5

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
    corr[np.abs(incidence) > np.radians(89)] = np.nan
    imgvals = imgvals * corr

    return imgvals


def create_image_from_grid(coords, imgvals, pix, inds, img_shape, n_neighbor=5, min_dist=25.):
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
