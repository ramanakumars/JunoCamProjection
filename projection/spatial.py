from pyproj.crs import CRS
from dataclasses import dataclass
import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class SpatialData:
    id: str
    image: np.ndarray
    crs: CRS
    x: np.ndarray
    y: np.ndarray

    def to_GeoTIFF(self, fname):
        # assume resolution is the same in both directions
        resolution = self.x[1] - self.x[0]

        transform = Affine.translation(self.x[0] - resolution / 2, self.y[-1] - resolution / 2) * Affine.scale(resolution, -resolution)
        with rasterio.open(fname, 'w', driver='GTiff', height=self.image.shape[0], width=self.image.shape[1], count=self.image.shape[2],
                           dtype=self.image.dtype, crs=self.crs, transform=transform) as dset:
            for i in range(self.image.shape[2]):
                dset.write(self.image[:, :, i], i + 1)
