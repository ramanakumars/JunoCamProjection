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
from .spatial import SpatialData, jupiter_crs
import healpy as hp
from pyproj import crs
from pyproj.transformer import Transformer


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

    def load_kernels(self, KERNEL_DATAFOLDER: str, offline: bool = False) -> None:
        """Get the kernels for the current spacecraft time and load them

        :param KERNEL_DATAFOLDER: path to the folder where kernels are stored
        :param offline: use the kernels stored locally (saves time by not scraping the NAIF servers)
        """
        self.kernels = []
        kernels = get_kernels(KERNEL_DATAFOLDER, self.start_utc, offline)
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

    def process(self, num_procs: int = 8, apply_correction: str = 'ls', minnaert_k: float = 1.05) -> None:
        """Processes the current image into a HEALPix map of a given resolution. Also applies lightning correction as needed.

        :param num_proces: number of cores to use for projection. Set to 1 to disable multithreading, defaults to 8
        :param apply_correction: the choice of lightning correction to apply. Choose betwen 'ls' for Lommel-Seeliger, 'minnaert' for Minnaert and 'none' for no correction, defaults to 'ls'
        :param minnaert_k: the index for Minnaert correction. Only used when apply_correction='minnaert', defaults to 1.25
        """
        print(f"Projecting {self.fname}")

        self.framedata.get_backplane(num_procs)

        self.apply_correction(apply_correction, minnaert_k)

    def apply_correction(self, correction_type: str, minnaert_k: float = 1.05) -> None:
        """Apply the requested illumination correction

        This function updates the framelet's `image` variable in-place and does not return a value

        :param apply_correction: the choice of lightning correction to apply. Choose betwen 'ls' for Lommel-Seeliger, 'minnaert' for Minnaert and 'none' for no correction, defaults to 'ls'
        :param minnaert_k: the index for Minnaert correction. Only used when apply_correction='minnaert', defaults to 1.25
        """
        if correction_type == 'ls':
            print("Applying Lommel-Seeliger correction")
            for frame in self.framedata.framelets:
                frame.image = apply_lommel_seeliger(frame.rawimg / frame.fluxcal, frame.incidence, frame.incidence)
        elif correction_type == 'minnaert':
            print("Applying Minnaert correction")
            for frame in self.framedata.framelets:
                frame.image = apply_minnaert(frame.rawimg / frame.fluxcal, frame.incidence, frame.incidence, k=minnaert_k)
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

    def project_to_pyproj(self, projection: crs.coordinate_operation.CoordinateOperation, resolution: float = 50, n_neighbor: int = 10, max_dist: float = 25):
        '''
        Convert the image to an arbitrary PyProj projection
        Note that we have to convert to a positive-east Sys III longitude so that PROJ does the right calculations

        :param resolution: the resolution of the final image in km/pixel
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :returns: the image in LAEA projection (shape: ny, nx, 3)
        '''
        transformer = Transformer.from_crs(jupiter_crs, projection)
        inv_transformer = Transformer.from_crs(projection, jupiter_crs)

        # find the locations in the image where we have valid data
        mask = np.isfinite(self.framedata.longitude)
        xx = np.zeros_like(self.framedata.longitude)
        yy = np.zeros_like(xx)

        # and calculate the distance between those points and the center of the image
        xx[mask], yy[mask] = transformer.transform(360 - self.framedata.longitude[mask], self.framedata.latitude[mask])

        # get the new camera grid with the given resolution
        # we are essentially constructing a field in LAEA projection
        xx_grid = np.arange(np.nanmin(xx), np.nanmax(xx), resolution * 1e3)
        yy_grid = np.arange(np.nanmin(yy), np.nanmax(yy), resolution * 1e3)

        XX, YY = np.meshgrid(xx_grid, yy_grid)

        # get the corresponding lat/lon grid for the image
        lon_grid, lat_grid = inv_transformer.transform(XX.flatten(), YY.flatten())

        # get the image extents in pixel coordinate space
        # clip half a pixel to avoid edge artifacts
        x0 = np.nanmin(self.framecoords[:, :, 0]) + 0.5
        x1 = np.nanmax(self.framecoords[:, :, 0]) - 0.5
        y0 = np.nanmin(self.framecoords[:, :, 1]) + 0.5
        y1 = np.nanmax(self.framecoords[:, :, 1]) - 0.5

        extents = np.array([x0, x1, y0, y1])
        pix = np.nan * np.zeros((lon_grid.size, 2))
        et = self.framedata.tmid

        # get the locations on the image where we have data
        get_pixel_from_coords_c(np.radians(360 - lon_grid.flatten()), np.radians(lat_grid.flatten()), lon_grid.size, et, extents, pix)

        inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
        pix_masked = pix[inds]
        pixel_inds = np.asarray(range(lon_grid.size))[inds]

        mapi = create_image_from_grid(self.framecoords, self.imagevalues, pixel_inds, pix_masked, lon_grid.shape, n_neighbor=n_neighbor, max_dist=max_dist)

        # finally, reshape into the 2D array and return it
        return SpatialData(self.fname, mapi.reshape((yy_grid.size, xx_grid.size, 3)), projection, xx_grid, yy_grid, lon_grid, lat_grid)

    def project_to_laea(self, resolution: float = 100, n_neighbor: int = 10, max_dist: float = 25) -> SpatialData:
        '''
        Convert the image to a Lambert Azimuthal Equal Area projection centered at the sub-spacecraft location.
        Note that we have to convert to a positive-east Sys III longitude so that PROJ does the right calculations

        :param resolution: the resolution of the final image in km/pixel
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :returns: the image in LAEA projection (shape: ny, nx, 3)
        '''
        # project with the sub-spacecraft location directly at the center
        latitude = self.framedata.sclat
        longitude = self.framedata.sclon

        # get the coordinate transformation from Cylindrical -> LAEA
        laea = crs.coordinate_operation.LambertAzimuthalEqualAreaConversion(latitude_natural_origin=latitude, longitude_natural_origin=360 - longitude)
        jupiter_laea = crs.ProjectedCRS(laea, 'Jupiter LAEA', crs.coordinate_system.Cartesian2DCS(), jupiter_crs)

        return self.project_to_pyproj(jupiter_laea, resolution, n_neighbor, max_dist)

    def project_to_mercator(self, resolution: float = 100, n_neighbor: int = 10, max_dist: float = 25) -> SpatialData:
        '''
        Convert the image to a Transverse Mercator projection centered at the sub-spacecraft location.
        Note that we have to convert to a positive-east Sys III longitude so that PROJ does the right calculations

        :param resolution: the resolution of the final image in km/pixel
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :returns: the image in Tranverse Mercator projection (shape: ny, nx, 3)
        '''
        # project with the sub-spacecraft location directly at the center
        latitude = self.framedata.sclat
        longitude = self.framedata.sclon

        # get the coordinate transformation from Cylindrical -> TMerc
        mercator = crs.coordinate_operation.TransverseMercatorConversion(latitude_natural_origin=latitude, longitude_natural_origin=360 - longitude)
        jupiter_mercator = crs.ProjectedCRS(mercator, 'Jupiter Mercator', crs.coordinate_system.Cartesian2DCS(), jupiter_crs)

        return self.project_to_pyproj(jupiter_mercator, resolution, n_neighbor, max_dist)

    def project_to_az_eqdist(self, resolution: float = 100, n_neighbor: int = 10, max_dist: float = 25) -> SpatialData:
        '''
        Convert the image to a Azimuthal Equidistant Projection centered at the sub-spacecraft location.
        Note that we have to convert to a positive-east Sys III longitude so that PROJ does the right calculations

        :param resolution: the resolution of the final image in km/pixel
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :returns: the image in Azimuthal Equidistant projection (shape: ny, nx, 3)
        '''
        # project with the sub-spacecraft location directly at the center
        latitude = self.framedata.sclat
        longitude = self.framedata.sclon

        # get the coordinate transformation from Cylindrical -> TMerc
        az_eqdist = crs.coordinate_operation.AzimuthalEquidistantConversion(latitude_natural_origin=latitude, longitude_natural_origin=360 - longitude)
        jupiter_az_eqdist = crs.ProjectedCRS(az_eqdist, 'Jupiter Azimuthal Equidistant', crs.coordinate_system.Cartesian2DCS(), jupiter_crs)

        return self.project_to_pyproj(jupiter_az_eqdist, resolution, n_neighbor, max_dist)

    def project_to_cylindrical(self, resolution: float = 50, n_neighbor: int = 10, max_dist: float = 25) -> SpatialData:
        '''
        Convert the image to a Cylindrical projection
        Note that we have to convert to a positive-east Sys III longitude so that PROJ does the right calculations

        :param resolution: the resolution of the final image in pixels/degree
        :param n_neighbor: the number of nearest neighbours to use for interpolating. Increase to get more details at the cost of performance, defaults to 5
        :param max_dist: the largest distance between neighbours to use for interpolation, defaults to 25 pixels

        :returns: the image in cylindrical projection (shape: nlat, nlon, 3)
        '''

        # get the coordinate transformation from Cylindrical -> TMerc
        eqcyl = crs.coordinate_operation.EquidistantCylindricalConversion()
        jupiter_cyl = crs.ProjectedCRS(eqcyl, 'Jupiter Cylindrical', crs.coordinate_system.Cartesian2DCS(), jupiter_crs)

        return self.project_to_pyproj(jupiter_cyl, resolution, n_neighbor, max_dist)

    def project_to_cylindrical_fullglobe(self, resolution: float = 50, n_neighbor: int = 10, max_dist: float = 25) -> SpatialData:
        # get the coordinate transformation from Cylindrical -> TMerc
        eqcyl = crs.coordinate_operation.EquidistantCylindricalConversion()
        projection = crs.ProjectedCRS(eqcyl, 'Jupiter Cylindrical', crs.coordinate_system.Cartesian2DCS(), jupiter_crs)

        # get the corresponding lat/lon grid for the image
        lon_grid = np.linspace(0, 360, resolution * 360 + 1, endpoint=True)
        lat_grid = np.linspace(-90, 90, resolution * 180 + 1, endpoint=True)

        LON, LAT = np.meshgrid(lon_grid, lat_grid)

        # get the image extents in pixel coordinate space
        # clip half a pixel to avoid edge artifacts
        x0 = np.nanmin(self.framecoords[:, :, 0]) + 0.5
        x1 = np.nanmax(self.framecoords[:, :, 0]) - 0.5
        y0 = np.nanmin(self.framecoords[:, :, 1]) + 0.5
        y1 = np.nanmax(self.framecoords[:, :, 1]) - 0.5

        extents = np.array([x0, x1, y0, y1])
        pix = np.nan * np.zeros((LON.size, 2))
        et = self.framedata.tmid

        # get the locations on the image where we have data
        get_pixel_from_coords_c(np.radians(360 - LON.flatten()), np.radians(LAT.flatten()), LON.size, et, extents, pix)

        inds = np.where(np.isfinite(pix[:, 0] * pix[:, 1]))[0]
        pix_masked = pix[inds]
        pixel_inds = np.asarray(range(LON.size))[inds]

        mapi = create_image_from_grid(self.framecoords, self.imagevalues, pixel_inds, pix_masked, LON.shape, n_neighbor=n_neighbor, max_dist=max_dist)

        # finally, reshape into the 2D array and return it
        return SpatialData(self.fname, mapi.reshape((lat_grid.size, lon_grid.size, 3)), projection, lon_grid, lat_grid, lon_grid, lat_grid)

    @classmethod
    def load(cls, infile: str, kerneldir: str = './', offline=False):
        '''Load the object from a netCDF file

        :param infile: path to the input .nc file
        :param kerneldir: Path to folder where SPICE kernels will be stored, defaults to "./"
        :param offline: use the kernels stored locally (saves time by not scraping the NAIF servers)

        :return: the Projector object with the loaded data and backplane information
        '''

        self = cls.__new__(cls)

        with nc.Dataset(infile, 'r') as indata:
            self.fname = indata.id
            self.start_utc = indata.start_utc
            self.load_kernels(kerneldir, offline)

            self.framedata = FrameletData.from_file(indata.start_et, indata.sub_lat, indata.sub_lon, indata.frame_delay, indata.exposure,
                                                    indata.variables['rawimage'][:].astype(float), indata.variables['latitude'][:].astype(float), indata.variables['longitude'][:].astype(float),
                                                    indata.variables['incidence'][:].astype(float), indata.variables['emission'][:].astype(float),
                                                    indata.variables['fluxcal'][:].astype(float), indata.variables['coords'][:].astype(float))
            self.jitter = indata.jitter
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

            latitude = outdata.createVariable('latitude', 'float32', ('frames', 'colors', 'height', 'width'))
            longitude = outdata.createVariable('longitude', 'float32', ('frames', 'colors', 'height', 'width'))
            incidence = outdata.createVariable('incidence', 'float32', ('frames', 'colors', 'height', 'width'))
            emission = outdata.createVariable('emission', 'float32', ('frames', 'colors', 'height', 'width'))
            image = outdata.createVariable('rawimage', 'float32', ('frames', 'colors', 'height', 'width'))
            fluxcal = outdata.createVariable('fluxcal', 'float32', ('frames', 'colors', 'height', 'width'))
            coords = outdata.createVariable('coords', 'float32', ('frames', 'colors', 'height', 'width', 'xy'))

            outdata.id = self.fname
            outdata.start_utc = self.start_utc
            outdata.start_et = spice.str2et(self.start_utc)
            outdata.frame_delay = self.framedata.frame_delay
            outdata.jitter = self.jitter
            outdata.sub_lat = self.framedata.sclat
            outdata.sub_lon = self.framedata.sclon
            outdata.exposure = self.framedata.exposure

            rawimg = np.stack([frame.rawimg for frame in self.framedata.framelets], axis=0).reshape((self.framedata.nframes, 3, 128, 1648))
            fcal = np.stack([frame.fluxcal for frame in self.framedata.framelets], axis=0).reshape((self.framedata.nframes, 3, 128, 1648))

            latitude[:] = self.framedata.latitude[:]
            longitude[:] = self.framedata.longitude[:]
            incidence[:] = self.framedata.incidence[:]
            emission[:] = self.framedata.emission[:]
            image[:] = rawimg[:]
            fluxcal[:] = fcal[:]
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
    corr[np.abs(incidence) > np.radians(89.9)] = np.nan
    imgvals = imgvals * corr
    imgvals[~np.isfinite(imgvals)] = 0.

    return imgvals


def apply_minnaert(imgvals: np.ndarray, incidence: np.ndarray, emission: np.ndarray, k: float = 1.05, trim=-8) -> np.ndarray:
    """Apply the Minnaert illumination correction

    :param imgvals: the raw image values
    :param incidence: the incidence angles (in radians) for each pixel in `imgvals`
    :param emission: the emission angles (in radians) for each pixel in `imgvals`
    :param minnaert_k: the index for Minnaert correction, defaults to 0.95

    :return: the corrected image values with the same shape as `imgvals`
    """
    # apply Minnaert correction
    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    corr = (mu ** k) * (mu0 ** (k - 1))
    # log(mu * mu0) < -4 is usually pretty noisy
    corr[np.log(np.cos(incidence) * np.cos(emission)) < trim] = np.inf
    imgvals = imgvals / corr
    imgvals[~np.isfinite(imgvals)] = 0.

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
