import numpy as np
from .globals import FRAME_WIDTH, FRAME_HEIGHT, initializer
from .cython_utils import project_midplane_c
from .camera_funcs import CameraModel
import spiceypy as spice
import matplotlib.pyplot as plt
from skimage import io
import os
import sys
import tqdm
import multiprocessing


class FrameletData:
    """Holds information and methods pertaining to all the framelets in the image
    """
    def __init__(self, metadata: dict, imgfolder: str):
        """Initialize the structure by reading in the metadata and separating the framelets in the image

        :param metadata: input image metadata
        :param imgfolder: path to the folder where the image is stored
        """
        self.nframelets = int(metadata["LINES"] / FRAME_HEIGHT)
        # number of RGB frames
        self.nframes = int(self.nframelets / 3)

        print(f'Found {self.nframes} RGB frames')

        start_utc = metadata["START_TIME"]
        self.start_et = spice.str2et(start_utc)
        fname = metadata["FILE_NAME"].replace("-raw.png", "")
        fullimg = plt.imread(imgfolder + "%s-raw.png" % fname)

        # ignore color channels in the image
        if len(fullimg.shape) == 3:
            fullimg = fullimg.mean(axis=-1)

        self.sclat = float(metadata["SUB_SPACECRAFT_LATITUDE"])
        self.sclon = float(metadata["SUB_SPACECRAFT_LONGITUDE"])

        # get the exposure and convert to seconds
        self.exposure = (
            float(metadata["EXPOSURE_DURATION"].replace("<ms>", "").strip()) / 1.0e3
        )

        intframe_delay = metadata["INTERFRAME_DELAY"].split(" ")
        frame_delay = float(intframe_delay[0])

        # flatfield and gain from Brian Swift's GitHub
        # (https://github.com/BrianSwift/JunoCam/tree/master/Juno3D)
        self.flatfield = np.array(
            io.imread(
                os.path.dirname(__file__) + "/cal/flatFieldsSmooth12to16.tiff"
            )
        )
        self.flatfield[self.flatfield == 0] = 1.0

        self.framelets = []
        for n in tqdm.tqdm(range(self.nframes), desc='Decompanding'):
            for c in range(3):
                startrow = 3 * FRAME_HEIGHT * n + c * FRAME_HEIGHT
                endrow = 3 * FRAME_HEIGHT * n + (c + 1) * FRAME_HEIGHT
                flati = self.flatfield[(c * FRAME_HEIGHT): ((c + 1) * FRAME_HEIGHT), :]

                img = fullimg[startrow:endrow, :]
                framei = decompand(img) / 16384
                framei = framei / (flati * self.exposure)

                self.framelets.append(Framelet(self.start_et, frame_delay, n, c, framei))

        self.fullimg = np.concatenate([frame.rawimg for frame in self.framelets], axis=0)

    def get_backplane(self, num_procs: int) -> None:
        """Retrieve backplane information for all framelets (i.e., get pixel coordinates in the mid-plane frame and also incidence/emission angles)

        :param num_procs: number of processors to use for multi-threaded processing
        """
        tmid = self.tmid

        if num_procs > 1:
            inpargs = [(frame, tmid) for frame in self.framelets]
            with multiprocessing.Pool(
                processes=num_procs, initializer=initializer
            ) as pool:
                try:
                    with tqdm.tqdm(inpargs, desc='Projecting framelets') as pbar:
                        self.framelets = pool.starmap(Framelet.project_to_midplane, pbar)
                        pool.close()
                except KeyboardInterrupt:
                    pool.terminate()
                    sys.exit()
                finally:
                    pool.join()
        else:
            for frame in tqdm.tqdm(self.framelets):
                frame.project_to_midplane(tmid)

    def update_jitter(self, jitter: float) -> None:
        """Update the jitter in the spacecraft clock for each framelet

        :param jitter: the new jitter value in seconds
        """
        self.jitter = jitter
        for framelet in self.framelets:
            framelet.jitter = jitter

    @property
    def tmid(self) -> float:
        """The mid-plane clock time"""
        return np.mean([frame.et for frame in self.framelets])

    @property
    def image(self) -> np.ndarray:
        """The concatenated illumination-corrected image in all the framelets"""
        return np.stack([frame.image for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

    @property
    def coords(self) -> np.ndarray:
        """The concatenated coordinate of each pixel in the mid-plane frame for each pixel in the camera frame"""
        return np.stack([frame.coords for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH, 2))

    @property
    def emission(self) -> np.ndarray:
        """The concatenated emission angles for each pixel in the mid-plane frame"""
        return np.stack([frame.emission for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

    @property
    def incidence(self) -> np.ndarray:
        """The concatenated incidence angles for each pixel in the mid-plane frame"""
        return np.stack([frame.incidence for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

    @property
    def longitude(self) -> np.ndarray:
        """The concatenated Sys III longitude values for each pixel in the mid-plane frame"""
        return np.stack([frame.lon for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

    @property
    def latitude(self) -> np.ndarray:
        """The concatenated planetographic latitude value for each pixel in the mid-plane frame"""
        return np.stack([frame.lat for frame in self.framelets], axis=0).reshape((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))


class Framelet:
    """Holds information and methods for a single framelet in the JunoCam image"""
    def __init__(self, start_et: float, frame_delay: float, frame_no: int, color: int, img: np.ndarray):
        """Initialize the framelet with relevant spacecraft data and camera information

        :param start_et: the spacecraft clock time for the start of the exposure in ET
        :param frame_delay: the inter-frame delay for this camera
        :param frame_no: the index of the 3-color subframe (i.e., the 4th framelet is blue frame #2)
        :param color: the filter for this framelet (i.e., 0 for blue, 1 for green and 2 for red)
        """
        self.start_et = start_et
        self.frame_delay = frame_delay
        self.frame_no = frame_no
        self.color = color
        self.rawimg = img
        self.jitter = None
        self.camera = CameraModel(color)

    @property
    def et(self) -> float:
        """Get the observation time for this framelet"""
        jitter = 0 if self.jitter is None else self.jitter
        return (
            self.start_et + self.camera.time_bias + jitter + (self.frame_delay + self.camera.iframe_delay) * self.frame_no
        )

    def project_to_midplane(self, tmid: float):
        """Get the backplane information at the mid-plane frame for this framelet. Returns self for multi-processing

        :param tmid: the spacecraft clock at the mid-plane frame
        """
        coords = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2))
        lat = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        lon = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        incidence = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        emission = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
        fluxcal = np.nan * np.zeros((FRAME_HEIGHT, FRAME_WIDTH))

        project_midplane_c(self.et, self.color, tmid, lon, lat, incidence, emission, coords, fluxcal)

        # geometry correction
        self.fluxcal = fluxcal
        self.coords = coords
        self.image = self.rawimg / self.fluxcal
        self.lat = lat
        self.lon = lon

        # emission and incidence angles for lightning correction
        self.emission = emission
        self.incidence = incidence

        return self


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


def decompand(image: np.ndarray) -> np.ndarray:
    """Decompands the image from the 8-bit in the public release
    to the original 12-bit shot by JunoCam

    :param image: 8-bit input image

    :return: Original 12-bit image
    """
    data = np.array(255 * image, dtype=np.uint8)
    ny, nx = data.shape

    def get_sqrt(x: float) -> float:
        return SQROOT[x]
    v_get_sqrt = np.vectorize(get_sqrt)

    data2 = v_get_sqrt(data)

    return data2
