import numpy as np
from .globals import FRAME_WIDTH, FRAME_HEIGHT


class FrameletData:
    def __init__(self, nframes):
        self.nframes = nframes

        # store the pixel coordinates for the midpoint plane
        self.coords = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH, 2))

        # corresponding pixel image values
        self.image = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        self.rawimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        # latitudes/longitudes
        self.lat = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        self.lon = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        # and the backplane information
        self.emission = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        self.incidence = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

        # and the calibration
        self.fluxcal = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))

    def update_image(self, new_image):
        assert new_image.shape == self.image.shape, f"New image must be the same shape as the old one. Got {new_image.shape} instead of {self.image.shape}"

        # first back up the original
        self.image_old = self.image
        # then store the new image data
        self.image[:] = new_image[:]
