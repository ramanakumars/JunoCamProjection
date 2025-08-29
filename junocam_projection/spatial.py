from pyproj.crs import CRS
from dataclasses import dataclass
import numpy as np
import rasterio
import os
import cartopy.crs as ccrs
from pyproj.transformer import Transformer
from rasterio.transform import Affine
from pyproj import crs

# set expected types to fix the NotImplementedError
ccrs._CRS._expected_types = ("Projected CRS", "Derived Projected CRS")

# define the ellipsoid and datum for Jupiter
jupiter = crs.datum.CustomEllipsoid(
    "Jupiter", semi_major_axis=71492e3, semi_minor_axis=66854e3
)
primem = crs.datum.CustomPrimeMeridian(longitude=0, name="Jupiter Prime Meridian")
jupiter_datum = crs.datum.CustomDatum(
    "Jupiter", ellipsoid=jupiter, prime_meridian=primem
)

# this is the base cylindrical projection for Jupiter
jupiter_crs = crs.GeographicCRS("Jupiter", datum=jupiter_datum)


@dataclass
class SpatialData:
    id: str
    image: np.ndarray
    crs: CRS
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    def to_GeoTIFF(self, fname):
        """Save the data to GeoTIFF with the right extents

        :param fname: path to the GeoTIFF
        """

        # assume resolution is the same in both directions
        resolution = self.x[1] - self.x[0]

        transform = Affine.translation(
            self.x[0] - resolution / 2, self.y[-1] - resolution / 2
        ) * Affine.scale(resolution, -resolution)
        with rasterio.open(
            fname,
            "w",
            driver="GTiff",
            height=self.image.shape[0],
            width=self.image.shape[1],
            count=self.image.shape[2],
            dtype=self.image.dtype,
            crs=self.crs,
            transform=transform,
        ) as dset:
            for i in range(self.image.shape[2]):
                dset.write(self.image[:, :, i], i + 1)

    @classmethod
    def from_GeoTIFF(cls, fname):
        """
        Read in a GeoTIFF

        :param fname: path to GeoTIFF file
        """
        with rasterio.open(fname, "r") as infile:
            bounds = infile.bounds
            crs = infile.crs
            image = np.transpose(infile.read(), (1, 2, 0))

        x = (
            bounds.left
            + np.arange(image.shape[1]) * (bounds.right - bounds.left) / image.shape[1]
        )
        y = (
            bounds.bottom
            + np.arange(image.shape[0]) * (bounds.top - bounds.bottom) / image.shape[0]
        )

        transformer = Transformer.from_crs(crs, jupiter_crs)

        xx_grid, yy_grid = np.meshgrid(x, y)

        XX, YY = np.meshgrid(xx_grid, yy_grid)

        # get the corresponding lat/lon grid for the image
        lon_grid, lat_grid = transformer.transform(XX.flatten(), YY.flatten())

        return cls(os.path.splitext(fname)[0], image, crs, x, y, lon_grid, lat_grid)

    @property
    def cartopy_crs(self):
        """
        Get the CRS for cartopy for plotting

        :return: the Cartopy CRS object
        """
        data_crs = ccrs.Projection.from_wkt(self.crs.to_wkt())
        data_crs.bounds = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
        data_crs.proj4_init = False

        return data_crs
