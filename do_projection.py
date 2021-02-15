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
from projection_funcs import Projector, map_project, map_project_multi
import argparse

parser = argparse.ArgumentParser(description="Map projection of a single JunoCam image")
parser.add_argument('id', nargs='+', type=int, help='4-digit ID of the image')
parser.add_argument('-np', '--num_procs', type=int, default=1, help='number of processes to use')
parser.add_argument('-res', '--resolution', type=int, default=50, help='pixels per degree')
parser.add_argument('-mos', '--mosaic', action='store_true', default=False, help='mosaic all frames together')

args = parser.parse_args()

ids = args.id

if(len(ids) > 0):
    print("Projecting %d files with %d processors"%(len(ids), args.num_procs))
else:
    print("Projecting %d with %d processors"%(ids[0], args.num_procs))

if not args.mosaic:
    print("Saving each file with a resolution of %d pixels per degree"%(args.resolution))
else:
    print("Saving a mosaic with a resolution of %d pixels per degree"%(args.resolution))

fnames = []

for idi in ids:
    projector = Projector("ImageSet/", "DataSet/%d-Metadata.json"%idi)
    projector.process(args.num_procs)

    fnames.append(projector.fname+".nc")
    
    if not args.mosaic:
        newlon = np.arange(projector.lonmin, projector.lonmax, 1./args.resolution)
        newlat = np.arange(projector.latmin, projector.latmax, 1./args.resolution)
        print("Mosaic size: %d x %d"%(newlon.size, newlat.size))
        IMG, mask = map_project(newlon, newlat, "%s.nc"%projector.fname, gamma=1.5, save=True)
if(len(fnames) > 1):
    if(args.mosaic):
        print("Mosaicing")
        newlon, newlat, IMG = map_project_multi(fnames, pixres=1./args.resolution)
