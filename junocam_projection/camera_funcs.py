import spiceypy as spice
import numpy as np


# filter ids for B, G and R
FILTERS = ["B", "G", "R"]
CAMERA_IDS = [-61501, -61502, -61503]


class CameraModel:
    """Holds the camera model and filter specific variables"""

    def __init__(self, filt: int):
        """Initialize the camera and load the corresponding variables from the SPICE kernel pool

        :param filt: The filter number (0 for blue, 1 for green and 2 for red)
        """
        self.filter = filt
        self.id = CAMERA_IDS[filt]

        # get the camera distortion data
        self.k1 = spice.gdpool("INS%s_DISTORTION_K1" % (self.id), 0, 32)[0]
        self.k2 = spice.gdpool("INS%s_DISTORTION_K2" % (self.id), 0, 32)[0]
        self.cx = spice.gdpool("INS%s_DISTORTION_X" % (self.id), 0, 32)[0]
        self.cy = spice.gdpool("INS%s_DISTORTION_Y" % (self.id), 0, 32)[0]
        self.flength = spice.gdpool("INS%s_FOCAL_LENGTH" % (self.id), 0, 32)[0]
        self.psize = spice.gdpool("INS%s_PIXEL_SIZE" % (self.id), 0, 32)[0]
        self.f1 = self.flength / self.psize

        # get the timing bias
        self.time_bias = spice.gdpool("INS%s_START_TIME_BIAS" % self.id, 0, 32)[0]
        self.iframe_delay = spice.gdpool("INS%s_INTERFRAME_DELTA" % self.id, 0, 32)[0]

    def pix2vec(self, px: list[float]) -> np.ndarray:
        """Convert from pixel coordinate to vector in the JUNO_JUNOCAM reference frame. See: https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/ik/juno_junocam_v03.ti

        :param px: x and y position of pixel centers in the camera

        :return: vector in the JUNO_JUNOCAM reference frame
        """
        camx = px[0] - self.cx
        camy = px[1] - self.cy
        cam = self.undistort([camx, camy])
        v = np.asarray([cam[0], cam[1], self.f1])
        return v

    def undistort(self, c: list[float]) -> tuple[float]:
        """Removes the barrel distortion in the JunoCam image

        :param c: x and y position of pixel centers in the camera

        :return: tuple containing the x- and y-position of the pixel after removing barrel distortion
        """
        xd, yd = c[0], c[1]
        for i in range(5):
            r2 = xd**2.0 + yd**2.0
            dr = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            xd = c[0] / dr
            yd = c[1] / dr
        return (xd, yd)

    def distort(self, c: list[float]) -> tuple[float]:
        """Adds barrel distortion to the image

        :param c: x and y position of undistorted pixel centers in the camera

        :return: x- and y- position of the pixel after adding barrel distortion
        """
        xd, yd = c[0], c[1]
        r2 = xd**2 + yd**2
        dr = 1 + self.k1 * r2 + self.k2 * r2 * r2
        xd *= dr
        yd *= dr
        return (xd, yd)

    def vec2pix(self, v: list[float]) -> tuple[float]:
        """Convert a vector in the JUNO_JUNOCAM reference frame to pixel coordinates on the plate

        :param v: vector in the JUNO_JUNOCAM reference frame

        :return: x- and y-center of the pixel in the plate
        """
        alpha = v[2] / self.f1
        cam = [v[0] / alpha, v[1] / alpha]
        cam = self.distort(cam)
        x = cam[0] + self.cx
        y = cam[1] + self.cy
        return (x, y)
