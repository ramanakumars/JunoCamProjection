import numpy as np
import json
import netCDF4 as nc
from skimage import feature
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import spiceypy as spice
import tqdm
from .cython_utils import furnish_c, get_pixel_from_coords_c
from .camera_funcs import CameraModel
from .spice_utils import get_kernels
from .frameletdata import FrameletData
import healpy as hp


class Projector:
    """
    The main projector class that determines the surface intercept points
    of each pixel in a JunoCam image
    """

    def __init__(self, imagefolder: str, meta: str, kerneldir: str = "./kernels/"):
        """Initializes the Projector

        :param imagefolder: Path to the folder containing the image files
        :param meta: Path to the metadata JSON file
        :param kerneldir: Path to folder where SPICE kernels will be stored, defaults to "./"
        """
        with open(meta, "r") as metafile:
            metadata = json.load(metafile)

        self.start_utc = metadata["START_TIME"]
        self.fname = metadata["FILE_NAME"].replace("-raw.png", "")

        print(f"Loading data for {self.fname}")

        # number of strips
        self.load_kernels(kerneldir)

        self.framedata = FrameletData(metadata, imagefolder)

        self.find_jitter(jitter_max=120)

    def load_kernels(self, KERNEL_DATAFOLDER: str) -> None:
        """Get the kernels for the current spacecraft time and load them

        :param KERNEL_DATAFOLDER: path to the folder where kernels are stored
        """
        self.kernels = []
        kernels = get_kernels(KERNEL_DATAFOLDER, self.start_utc)
        for kernel in kernels:
            furnish_c(kernel.encode("ascii"))
            spice.furnsh(kernel)
            self.kernels.append(kernel)

    def find_jitter(self, jitter_max: float = 25, threshold: float = 80, plot: bool = False) -> None:
        """Find the best jitter value to the spacecraft camera time


        :param jitter_max: Maximum value to search for the jitter [millseconds], defaults to 25
        :param threshold: percentile value to use to find the planet's limb, defaults to 80
        :param plot: boolean flag for whether to plot the intermediate steps for debugging, defaults to False
        """
        threshold = np.percentile(self.framedata.fullimg.flatten(), threshold)

        for nci in range(self.framedata.nframes * 3):
            # find whether the planet limb is in now
            # approximating this as the first time the planet is seen
            # in the image, which is generally true..
            ci = nci % 3

            frame = self.framedata.framelets[nci].rawimg
            eti = self.framedata.framelets[nci].et

            if np.sum(frame > threshold) > 5000:
                # make sure that the limb is also in the image
                # we want to find a frame where both the predicted limb
                # with no jitter and the actual limb are visible so that
                # the offset we determine is small
                cami = CameraModel(ci)
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
            limbs_jcam = self.get_limb(eti + jitter, cami)

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

        self.framedata.update_jitter(self.jitter)
        eti = self.framedata.framelets[nci].et

        limbs_jcam = self.get_limb(eti, cami)

        if plot:
            plt.plot(limbs_jcam[:, 0], limbs_jcam[:, 1], "g.", markersize=0.1)
            plt.show()

    def get_limb(self, eti: float, cami: CameraModel) -> np.ndarray:
        """Find the pixel positions of the planet's limb for a given camera frame at a specific time

        :param eti: spacecraft clock time in seconds
        :param cami: the camera which is used for observing

        :return: the pixel positions of the limb of the planet in the camera frame (shape: (npix, 2))
        """
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

    def process(self, nside: int = 512, num_procs: int = 8, apply_correction: str = 'ls', n_neighbor: int = 5, minneart_k: float = 1.05) -> np.ndarray:
        """Processes the current image into a HEALPix map of a given resolution. Also applies lightning correction as needed.

        :param nside: resolution of the HEALPix map. See https://healpy.readthedocs.io/en/latest/tutorial.html, defaults to 512
        :param num_proces: number of cores to use for projection. Set to 1 to disable multithreading, defaults to 8
        :param apply_correction: the choice of lightning correction to apply. Choose betwen 'ls' for Lommel-Seeliger, 'minneart' for Minneart and 'none' for no correction, defaults to 'ls'
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param minneart_k: the index for Minneart correction. Only used when apply_correction='minneart', defaults to 1.25

        :return: the HEALPix map of the projected image (shape: (npixels, 3), where npixels is the corresponding image size for a given n_side)
        """
        print(f"Projecting {self.fname} to a HEALPix grid with n_side={nside}")

        self.framedata.get_backplane(num_procs)

        self.apply_correction(apply_correction, minneart_k)

        map = self.project_to_healpix(nside, n_neighbor=n_neighbor)

        return map

    def apply_correction(self, correction_type: str, minneart_k: float = 1.05) -> None:
        """Apply the requested illumination correction

        This function updates the framelet's `image` variable in-place and does not return a value

        :param apply_correction: the choice of lightning correction to apply. Choose betwen 'ls' for Lommel-Seeliger, 'minneart' for Minneart and 'none' for no correction, defaults to 'ls'
        :param minneart_k: the index for Minneart correction. Only used when apply_correction='minneart', defaults to 1.25
        """
        if correction_type == 'ls':
            print("Applying Lommel-Seeliger correction")
            for frame in self.framedata.framelets:
                frame.image = apply_lommel_seeliger(frame.rawimg / frame.fluxcal, frame.incidence, frame.incidence)
        elif correction_type == 'minneart':
            print("Applying Minneart correction")
            for frame in self.framedata.framelets:
                frame.image = apply_minneart(frame.rawimg / frame.fluxcal, frame.incidence, frame.incidence, k=minneart_k)
        elif correction_type == 'none':
            print("Applying no correction")
            for frame in self.framedata.framelets:
                frame.image = frame.rawimg / frame.fluxcal

    @property
    def framecoords(self) -> np.ndarray:
        """Get the coordinates of each pixel in the current camera frame in the midplane frame
        """
        return np.transpose(self.framedata.coords, (1, 0, 2, 3, 4)).reshape(3, -1, 2)

    @property
    def imagevalues(self) -> np.ndarray:
        """Get the illumination corrected image values from each frame
        """
        return np.transpose(self.framedata.image, (1, 0, 2, 3)).reshape(3, -1)

    def project_to_healpix(self, nside: int, n_neighbor: int = 4, max_dist: int = 25) -> np.ndarray:
        """Project the current image to a HEALPix map

        :param nside: resolution of the HEALPix map. See https://healpy.readthedocs.io/en/latest/tutorial.html
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :return: the HEALPix map of the projected image (shape: (npixels, 3), where npixels is the corresponding image size for a given n_side)
        """
        # get the image extents in pixel coordinate space
        # clip half a pixel to avoid edge artifacts
        x0 = np.nanmin(self.framecoords[:, :, 0]) + 0.5
        x1 = np.nanmax(self.framecoords[:, :, 0]) - 0.5
        y0 = np.nanmin(self.framecoords[:, :, 1]) + 0.5
        y1 = np.nanmax(self.framecoords[:, :, 1]) - 0.5

        extents = np.array([x0, x1, y0, y1])

        # now we construct the map grid
        longrid, latgrid = hp.pix2ang(nside, list(range(hp.nside2npix(nside))), lonlat=True)

        pix = np.nan * np.zeros((longrid.size, 2))

        # this assumes the midplane mode where the coordinates
        # are using the Green band camera
        et = self.framedata.tmid

        # get the spacecraft location and transformation to and from the JUNOCAM coordinates
        get_pixel_from_coords_c(np.radians(longrid), np.radians(latgrid), longrid.size, et, extents, pix)

        # get the locations that JunoCam observed
        inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
        pix = pix[inds]
        pixel_inds = hp.ang2pix(nside, longrid[inds], latgrid[inds], lonlat=True)

        # finally, project the image onto the healpix grid
        m = create_image_from_grid(self.framecoords, self.imagevalues, pixel_inds, pix, longrid.shape, n_neighbor=n_neighbor, max_dist=max_dist)

        return m

    @classmethod
    def load(cls, infile: str, kerneldir: str = './'):
        '''Load the object from a netCDF file

        :param infile: path to the input .nc file
        :param kerneldir: Path to folder where SPICE kernels will be stored, defaults to "./"

        :return: the Projector object with the loaded data and backplane information
        '''

        self = cls.__new__(cls)

        with nc.Dataset(infile, 'r') as indata:
            self.fname = indata.id
            self.start_utc = indata.start_utc
            self.load_kernels(kerneldir)

            self.framedata = FrameletData.from_file(indata.start_et, indata.sub_lon, indata.sub_lat, indata.frame_delay, indata.exposure,
                                                    indata.variables['rawimage'][:], indata.variables['latitude'][:], indata.variables['longitude'][:],
                                                    indata.variables['incidence'][:], indata.variables['emission'][:],
                                                    indata.variables['fluxcal'][:], indata.variables['coords'][:])
            self.framedata.update_jitter(indata.jitter)

        return self

    def save(self, outfile: str) -> None:
        '''Save the projection data to a netCDF file

        :param outfile: path to the .nc file to save to
        '''
        with nc.Dataset(outfile, 'w') as outdata:
            outdata.createDimension('frames', self.framedata.nframes)
            outdata.createDimension('colors', 3)
            outdata.createDimension('width', 1648)
            outdata.createDimension('height', 128)
            outdata.createDimension('xy', 2)

            latitude = outdata.createVariable('latitude', 'float64', ('frames', 'colors', 'height', 'width'))
            longitude = outdata.createVariable('longitude', 'float64', ('frames', 'colors', 'height', 'width'))
            incidence = outdata.createVariable('incidence', 'float64', ('frames', 'colors', 'height', 'width'))
            emission = outdata.createVariable('emission', 'float64', ('frames', 'colors', 'height', 'width'))
            image = outdata.createVariable('rawimage', 'float64', ('frames', 'colors', 'height', 'width'))
            fluxcal = outdata.createVariable('fluxcal', 'float64', ('frames', 'colors', 'height', 'width'))
            coords = outdata.createVariable('coords', 'float64', ('frames', 'colors', 'height', 'width', 'xy'))

            outdata.id = self.fname
            outdata.start_utc = self.start_utc
            outdata.start_et = spice.str2et(self.start_utc)
            outdata.frame_delay = self.framedata.frame_delay
            outdata.jitter = self.jitter
            outdata.sub_lat = self.framedata.sclat
            outdata.sub_lon = self.framedata.sclon
            outdata.exposure = self.framedata.exposure

            rawimg = np.stack([frame.rawimg for frame in self.framedata.framelets], axis=0).reshape((self.framedata.nframes, 3, 128, 1648))
            fluxcal = np.stack([frame.fluxcal for frame in self.framedata.framelets], axis=0).reshape((self.framedata.nframes, 3, 128, 1648))

            latitude[:] = self.framedata.latitude[:]
            longitude[:] = self.framedata.longitude[:]
            incidence[:] = self.framedata.incidence[:]
            emission[:] = self.framedata.emission[:]
            image[:] = rawimg[:]
            fluxcal[:] = fluxcal[:]
            coords[:] = self.framedata.coords[:]


def apply_lommel_seeliger(imgvals: np.ndarray, incidence: np.ndarray, emission: np.ndarray) -> np.ndarray:
    '''Apply the Lommel-Seeliger correction for incidence

    :param imgvals: the raw image values
    :param incidence: the incidence angles (in radians) for each pixel in `imgvals`
    :param emission: the emission angles (in radians) for each pixel in `imgvals`

    :return: the corrected image values with the same shape as `imgvals`
    '''
    # apply Lommel-Seeliger correction
    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    corr = 1. / (mu + mu0)
    corr[np.abs(incidence) > np.radians(89)] = np.nan
    imgvals = imgvals * corr

    return imgvals


def apply_minneart(imgvals: np.ndarray, incidence: np.ndarray, emission: np.ndarray, k: float = 1.05) -> np.ndarray:
    """Apply the Minneart illumination correction

    :param imgvals: the raw image values
    :param incidence: the incidence angles (in radians) for each pixel in `imgvals`
    :param emission: the emission angles (in radians) for each pixel in `imgvals`
    :param minnaert_k: the index for Minneart correction, defaults to 0.95

    :return: the corrected image values with the same shape as `imgvals`
    """
    # apply Minneart correction
    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    corr = (mu ** k) * (mu0 ** (k - 1))
    # log(mu * mu0) < -4 is usually pretty noisy
    corr[np.log(np.cos(incidence) * np.cos(emission)) < -4] = np.inf
    imgvals = imgvals / corr

    return imgvals


def create_image_from_grid(coords: np.ndarray, imgvals: np.ndarray, inds: np.ndarray, pix: np.ndarray,
                           img_shape: tuple[int], n_neighbor: int = 5, max_dist: float = 25.):
    '''
        Reproject an irregular spaced image onto a regular grid from a list of coordinate
        locations and corresponding image values. This uses an inverse lookup-table defined
        by `pix`, where pix gives the coordinates in the original image where the corresponding
        pixel coordinate on the new image should be. The coordinate on the new image is given by
        the `inds`.

    :param coords: the pixel coordinates in the original image
    :param imgvals: the image values corresponding to coords
    :param inds: the coordinate on the new image where we need to interpolate
    :param pix: the coordinate in the original image corresponding to each pixel in inds
    :param img_shape: the shape of the new image
    :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
    :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

    :return: the interpolated new image of shape `img_shape` where every pixel at `inds` has corresponding values interpolated from `imgvals`
    '''
    nchannels, ncoords, _ = coords.shape

    newvals = np.zeros((nchannels, pix.shape[0]))
    print("Calculating image values at new locations")
    for n in range(nchannels):
        mask = np.isfinite(coords[n, :, 0] * coords[n, :, 1])
        neighbors = NearestNeighbors().fit(coords[n][mask])
        dist, indi = neighbors.kneighbors(pix, n_neighbor)
        weight = 1. / (dist + 1.e-16)
        weight = weight / np.sum(weight, axis=1, keepdims=True)
        weight[dist > max_dist] = 0.

        newvals[n, :] = np.sum(np.take(imgvals[n][mask], indi, axis=0) * weight, axis=1)

    IMG = np.zeros((*img_shape, nchannels))
    # loop through each point observed by JunoCam and assign the pixel value
    for k, ind in enumerate(tqdm.tqdm(inds, desc='Building image')):
        if len(img_shape) == 2:
            j, i = np.unravel_index(ind, img_shape)

            # do the weighted average for each filter
            for n in range(nchannels):
                IMG[j, i, n] = newvals[n, k]
        else:
            for n in range(nchannels):
                IMG[ind, n] = newvals[n, k]

    IMG[~np.isfinite(IMG)] = 0.

    return np.flip(IMG, axis=-1)
