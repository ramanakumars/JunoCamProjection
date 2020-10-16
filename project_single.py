'''
        Copyright (C) 2020 Ramanakumar Sankar

    This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
            the Free Software Foundation, either version 3 of the License, or
                (at your option) any later version.

    This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
                GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from projection_funcs import Projector, map_project
import cartopy.crs as ccrs

for idi in [8943]:
    projector = Projector("ImageSet/", "DataSet/%d-Metadata.json"%idi)
    projector.process(14)
    newlon = np.arange(projector.lonmin, projector.lonmax, 1./200.)
    newlat = np.arange(projector.latmin, projector.latmax, 1./200.)

    IMG, mask = map_project(newlon, newlat, "%s.nc"%projector.fname, save=True)
