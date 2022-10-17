import numpy as np
import json
import glob
import re
from skimage import feature
from sklearn.metrics import pairwise_distances
import os
import time
import matplotlib.pyplot as plt
import multiprocessing
import spiceypy as spice
import netCDF4 as nc
from skimage import io
from scipy.interpolate import griddata
import tqdm
from ftplib import FTP
from .globals import FRAME_HEIGHT, FRAME_WIDTH, NC_FOLDER, initializer
from .cython_utils import furnish_c, process_c
from .camera_funcs import CameraModel
import sys


# speed of light
c_light = 3.0e5

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

    def get_sqrt(x): return SQROOT[x]
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

        try:
            self.load_kernels(kerneldir)
        except Exception:
            self.download_kernels(kerneldir)

        self.re, _, self.rp = spice.bodvar(spice.bodn2c("JUPITER"), "RADII", 3)
        self.flattening = (self.re - self.rp) / self.re

        # calculate the start time
        self.start_et = spice.str2et(self.start_utc)

        self.savefolder = "%s_proj/" % self.fname

        self.jitter = 0.0

    def load_kernels(self, KERNEL_DATAFOLDER):
        """
        Determines and loads the required SPICE data kernels
        to project and process the current image

        Parameters
        ----------
        KERNEL_DATAFOLDER : string
            path to the location of the Juno kernels

        Raises
        ------
        AssertionError
            if no kernels were found for the date range
        """
        # find and load the kernels for a specific date
        iks = sorted(glob.glob(KERNEL_DATAFOLDER + "ik/juno_junocam_v*.ti"))
        cks = sorted(glob.glob(KERNEL_DATAFOLDER + "ck/juno_sc_*.bc"))
        spks1 = sorted(glob.glob(KERNEL_DATAFOLDER + "spk/spk_*.bsp"))
        spks2 = sorted(glob.glob(KERNEL_DATAFOLDER + "spk/jup*.bsp"))
        spks3 = sorted(glob.glob(KERNEL_DATAFOLDER + "spk/de*.bsp"))
        pcks = sorted(glob.glob(KERNEL_DATAFOLDER + "pck/pck*.tpc"))
        fks = sorted(glob.glob(KERNEL_DATAFOLDER + "fk/juno_v*.tf"))
        sclks = sorted(
            glob.glob(KERNEL_DATAFOLDER + "sclk/JNO_SCLKSCET.*.tsc")
        )
        lsks = sorted(glob.glob(KERNEL_DATAFOLDER + "lsk/naif*.tls"))

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
        kernels.append(pcks[-1])
        kernels.append(fks[-1])
        kernels.append(sclks[-1])
        kernels.append(lsks[-1])
        kernels.append(KERNEL_DATAFOLDER + "spk/juno_rec_orbit.bsp")
        kernels.append(KERNEL_DATAFOLDER + "spk/juno_pred_orbit.bsp")
        kernels.append(
            sorted(glob.glob(KERNEL_DATAFOLDER + "spk/juno_struct*.bsp"))[-1]
        )

        self.kernels = kernels
        for kernel in self.kernels:
            furnish_c(kernel.encode("ascii"))
            spice.furnsh(kernel)

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
        sclks = sorted(
            ftp.nlst("sclk/JNO_SCLKSCET.*.tsc")
        )
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
            print(kernel, kernel_local, flush=True)

            with open(kernel_local, 'wb') as kerfile:
                ftp.retrbinary(f"RETR {kernel}", kerfile.write)

            furnish_c(kernel_local.encode("ascii"))
            spice.furnsh(kernel_local)

            self.kernels.append(kernel_local)

    def process_n_c(self, inp):
        """
        Project a given frame and filter
        used in the multi-core version

        Parameters
        ----------
        inp : tuple
            framelet number and color index

        Outputs
        -------
        lats : numpy.ndarray
            array of latitudes in the same shape as the framelet (128,1648)
        lons : numpy.ndarray
            array of longitudes in the same shape as the framelet (128,1648)
        scloc : numpy.ndarray
            spacecraft position during the frame -- shape (3,)
        eti : double
            epoch of the spacecraft's observation for the given frame
        pixres : numpy.ndarray
            array of spatial resolutions of that pixel of shape (128,1648)
        """
        n, ci = inp
        try:
            self.latmin = 1000.0
            self.latmax = -1000.0
            self.lonmin = 1000.0
            self.lonmax = -1000.0

            cami = CameraModel(ci)
            eti = (
                self.start_et
                + cami.time_bias
                + self.jitter
                + (self.frame_delay + cami.iframe_delay) * n
            )
            """
                calculate the spacecraft position in the
                Jupiter reference frame
            """
            scloc, _ = spice.spkpos(
                "JUNO", eti, "IAU_JUPITER", "CN+S", "JUPITER"
            )

            """
                calculate the transformation from instrument
                to jupiter barycenter
            """
            cam2jup = spice.pxform("JUNO_JUNOCAM", "IAU_JUPITER", eti)

            lats = -1000.0 * np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            lons = -1000.0 * np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            # solar_corr = np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            incid = 1000.0 * np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            emis = 1000.0 * np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            flux_cal = np.zeros((FRAME_HEIGHT, FRAME_WIDTH))

            process_c(
                eti, ci, cam2jup.flatten(), lons, lats, incid, emis, flux_cal
            )

            # frame = decompand(frame[:])#*solar_corr[:]
            """
                find the resolution for each pixel and then calculate
                the finest resolution of the slice
            """

            dlats = np.gradient(lats)
            dlons = np.gradient(lons)

            dlat = (dlats[0] + dlats[1]) / 2.0
            dlon = (dlons[0] + dlons[1]) / 2.0
            dpix = np.sqrt(dlat**2.0 + dlon**2.0)

            if np.max(dpix) == 0.0:
                pixres = 0.0
            else:
                pixres = dpix[dpix > 0.0].min()
            return (lats, lons, scloc, eti, pixres, incid, emis, flux_cal)
        except Exception as e:
            raise (e)
            return

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

        for j, jitter in enumerate(jitter_list):
            eti = (
                self.start_et
                + cami.time_bias
                + jitter
                + (self.frame_delay + cami.iframe_delay) * n
            )

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
        refvec, _, _ = spice.subpnt(
            "INTERCEPT/ELLIPSOID",
            "JUPITER",
            eti,
            "IAU_JUPITER",
            "CN+S",
            "JUNO",
        )

        _, _, eplimbs, limbs_IAU = spice.limbpt(
            METHOD,
            "JUPITER",
            eti,
            "IAU_JUPITER",
            "CN+S",
            CORLOC,
            "JUNO",
            refvec,
            np.radians(0.1),
            3600,
            4.0,
            0.001,
            7200,
        )

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

    def process(self, num_procs=1):
        """
        Main driver for the projection. Determines line of sight
        intercepts for each pixel on the JunoCam image

        Parameters
        ----------
        num_procs : int
            Number of CPUs to use for multiprocessing [Default: 1]
        jitter : float
            offset in ms, to add to the start time to compensate for
            jitter [Default: 0]

        """
        print("%s" % self.fname)
        r = []

        print("Projecting framelets:")

        lat = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        lon = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        decompimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        rawimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        flux_cal = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        incid = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        emis = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        scloc = np.zeros((self.nframes, 3))
        et = np.zeros(self.nframes)

        # save these parameters to a NetCDF file so that we can plot it later
        if not os.path.exists(NC_FOLDER):
            os.mkdir(NC_FOLDER)

        self.find_jitter(jitter_max=120)

        # flatfield and gain from Brian Swift's GitHub
        # (https://github.com/BrianSwift/JunoCam/tree/master/Juno3D)
        flatfield = np.array(
            io.imread(
                os.path.dirname(__file__) + "/cal/flatFieldsSmooth12to16.tiff"
            )
        )

        inpargs = []
        self.image = np.zeros_like(self.fullimg)
        for n in range(self.nframes):
            for c in range(3):
                startrow = 3 * FRAME_HEIGHT * n + c * FRAME_HEIGHT
                endrow = 3 * FRAME_HEIGHT * n + (c + 1) * FRAME_HEIGHT

                framei = decompand(self.fullimg[startrow:endrow, :])
                flati = flatfield[
                    (c * FRAME_HEIGHT): ((c + 1) * FRAME_HEIGHT), :
                ]
                flati[flati == 0] = 1.0
                framei = framei / flati  # *gaini
                self.image[startrow:endrow, :] = framei / self.exposure

                inpargs.append((n, c))

        pixres = np.zeros(len(inpargs))

        pool = multiprocessing.Pool(
            processes=num_procs, initializer=initializer
        )
        try:
            r = pool.map_async(self.process_n_c, inpargs)
            pool.close()

            tasks = pool._cache[r._job]
            ninpt = len(inpargs)
            with tqdm.tqdm(total=ninpt) as pbar:
                while tasks._number_left > 0:
                    pbar.n = ninpt - tasks._number_left * tasks._chunksize
                    pbar.refresh()

                    time.sleep(0.1)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit()

        print()
        pool.join()

        results = r.get()
        for jj in range(len(inpargs)):
            (
                lati,
                loni,
                scloci,
                eti,
                pixres[jj],
                mu0i,
                mui,
                flux_cali,
            ) = results[jj]
            i, ci = inpargs[jj]
            startrow = 3 * FRAME_HEIGHT * i + ci * FRAME_HEIGHT
            endrow = 3 * FRAME_HEIGHT * i + (ci + 1) * FRAME_HEIGHT

            lat[i, ci, :, :] = lati
            lon[i, ci, :, :] = loni
            decompimg[i, ci, :, :] = self.image[startrow:endrow, :]
            rawimg[i, ci, :, :] = self.fullimg[startrow:endrow, :]
            incid[i, ci, :, :] = mu0i
            emis[i, ci, :, :] = mui
            flux_cal[i, ci, :, :] = flux_cali
            scloc[i, :] = scloci
            et[i] = eti

        pixres = pixres[pixres > 0.0]

        f = nc.Dataset("%s%s.nc" % (NC_FOLDER, self.fname), "w")

        f.createDimension("nframes", self.nframes)
        f.createDimension("ncolors", 3)
        f.createDimension("x", FRAME_WIDTH)
        f.createDimension("y", FRAME_HEIGHT)
        f.createDimension("xyz", 3)

        f.jitter = self.jitter * 1.0e3

        # create the NetCDF variables
        latVar = f.createVariable(
            "lat", "float32", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        lonVar = f.createVariable(
            "lon", "float32", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        imgVar = f.createVariable(
            "img", "float32", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        incVar = f.createVariable(
            "incidence", "float32", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        emiVar = f.createVariable(
            "emission", "float32", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        rawimgVar = f.createVariable(
            "rawimg", "uint8", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        fluximgVar = f.createVariable(
            "flux", "float64", ("nframes", "ncolors", "y", "x"), zlib=True
        )
        scVar = f.createVariable(
            "scloc", "float64", ("nframes", "xyz"), zlib=True
        )
        etVar = f.createVariable("et", "float64", "nframes", zlib=True)

        latVar[:] = lat[:]
        lonVar[:] = lon[:]
        imgVar[:] = decompimg[:]
        rawimgVar[:] = np.asarray(rawimg[:] * 255, dtype=np.uint8)
        fluximgVar[:] = decompimg[:] * flux_cal
        scVar[:] = scloc[:]
        etVar[:] = et[:]

        incVar[:] = incid
        emiVar[:] = emis

        f.close()

        mask = (lat != -1000.0) & (lon != -1000.0)
        self.lonmin = lon[mask].min()
        self.lonmax = lon[mask].max()
        self.latmin = lat[mask].min()
        self.latmax = lat[mask].max()

        print(
            "Extents - lon: %.3f %.3f lat: %.3f %.3f - lowest pixres: %.3f"
            " deg/pix"
            % (
                self.lonmin,
                self.lonmax,
                self.latmin,
                self.latmax,
                np.min(pixres),
            )
        )

    def project_to_midplane(self, num_procs=8):
        """
        Main driver for the projection to a camera frame. Projects
        a full frame onto a camera view of the planet

        Parameters
        ----------
        num_procs : int
            Number of CPUs to use for multiprocessing [Default: 1]
        """
        print("%s" % self.fname)

        decompimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        rawimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        scloc = np.zeros((self.nframes, 3))
        et = np.zeros((self.nframes, 3))

        # save these parameters to a NetCDF file so that we can plot it later
        if not os.path.exists(NC_FOLDER):
            os.mkdir(NC_FOLDER)

        self.find_jitter(jitter_max=120)

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

        for n in range(self.nframes):
            for c in range(3):
                cami = CameraModel(c)
                et[n, c] = self.start_et + cami.time_bias + self.jitter +\
                    (self.frame_delay + cami.iframe_delay) * n

        for n in tqdm.tqdm(range(self.nframes), desc='Decompanding'):
            for c in range(3):
                cami = CameraModel(c)
                startrow = 3 * FRAME_HEIGHT * n + c * FRAME_HEIGHT
                endrow = 3 * FRAME_HEIGHT * n + (c + 1) * FRAME_HEIGHT

                framei = decompand(self.fullimg[startrow:endrow, :])/16384

                flati = flatfield[
                    (c * FRAME_HEIGHT): ((c + 1) * FRAME_HEIGHT), :
                ]
                framei = framei / flati  # *gaini
                self.image[startrow:endrow, :] = framei / self.exposure

                inpargs.append((n, c, np.mean(et)))

        with multiprocessing.Pool(
            processes=num_procs, initializer=initializer
        ) as pool:
            print("Projecting framelets:")
            try:
                r = pool.map_async(self._project_to_midplane, inpargs)
                pool.close()

                tasks = pool._cache[r._job]
                ninpt = len(inpargs)
                with tqdm.tqdm(total=ninpt) as pbar:
                    while tasks._number_left > 0:
                        pbar.n = ninpt - tasks._number_left * tasks._chunksize
                        pbar.refresh()

                        time.sleep(0.1)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                sys.exit()

            print()
            pool.join()

        coords = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH, 2))
        imgvals = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        results = r.get()

        # fetch the coordinates and the image values
        for jj in range(len(inpargs)):
            coordsi = results[jj]
            i, ci = inpargs[jj][:2]
            startrow = 3 * FRAME_HEIGHT * i + ci * FRAME_HEIGHT
            endrow = 3 * FRAME_HEIGHT * i + (ci + 1) * FRAME_HEIGHT

            decompimg[i, ci, :, :] = self.image[startrow:endrow, :]
            rawimg[i, ci, :, :] = self.fullimg[startrow:endrow, :]

            coords[i, ci, :] = coordsi
            imgvals[i, ci, :] = decompimg[i, ci, :, :]

        coords = np.transpose(coords, (1, 0, 2, 3, 4)).reshape(3, -1, 2)
        imgvals = np.transpose(imgvals, (1, 0, 2, 3)).reshape(3, -1)

        # get the image extents
        x0 = np.nanmin(coords[:, :, 0])
        x1 = np.nanmax(coords[:, :, 0])
        y0 = np.nanmin(coords[:, :, 1])
        y1 = np.nanmax(coords[:, :, 1])

        xx, yy = np.meshgrid(np.arange(x0, x1, 0.5), np.arange(y0, y1, 0.5))

        # project onto a RGB image
        img_RGB = np.zeros((*xx.shape, 3))

        for c in range(3):
            coordsi = coords[c, :]
            x, y = coordsi[:, 0], coordsi[:, 1]
            mask = (np.isfinite(x)) & (np.isfinite(y))
            vals = imgvals[c, :][mask]

            x, y = x[mask], y[mask]
            img_RGB[:, :, 2-c] = griddata((x, y), vals,
                                          (xx, yy), method='cubic',
                                          fill_value=0)

        return img_RGB

    def _project_to_midplane(self, inpargs):
        n, c, tmid = inpargs
        ci = CameraModel(c)
        eti = self.start_et + ci.time_bias + self.jitter +\
            (self.frame_delay + ci.iframe_delay) * n

        pxfrm_mid = spice.pxfrm2('JUNO_JUNOCAM', 'JUNO_JUNOCAM', eti,
                                 tmid)
        pxfrm_iau = spice.pxform('JUNO_JUNOCAM', 'IAU_JUPITER', tmid)

        coords = np.nan*np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2))
        for j in range(FRAME_HEIGHT):
            for i in range(FRAME_WIDTH):
                veci = ci.pix2vec([i, j])

                vec_transformed = np.dot(
                    pxfrm_mid, veci
                )
                vec_iau = np.dot(
                    pxfrm_iau, vec_transformed
                ).reshape(3)

                with spice.no_found_check():
                    _, _, _, found = spice.sincpt('Ellipsoid', 'JUPITER',
                                                  tmid, 'IAU_JUPITER',
                                                  'CN+S', 'JUNO',
                                                  'IAU_JUPITER',
                                                  vec_iau)

                if found:
                    newx, newy = CameraModel(1).vec2pix(
                        vec_transformed
                    )
                else:
                    newx = newy = np.nan

                coords[j, i, :] = [newx, newy]
        return coords
