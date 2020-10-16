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
from projection_funcs import Projector, map_project
import argparse

parser = argparse.ArgumentParser(description="Map projection of a single JunoCam image")
parser.add_argument('id', nargs='+', type=int, help='4-digit ID of the image')
parser.add_argument('-np', '--num_procs', type=int, default=1, help='number of processes to use')

args = parser.parse_args()

ids = args.id

if(len(ids) > 0):
    print("Projecting %d files with %d processors"%(len(ids), args.num_procs))
else:
    print("Projecting %d with %d processors"%(ids[0], args.num_procs))

for idi in ids:
    projector = Projector("ImageSet/", "DataSet/%d-Metadata.json"%idi)
    projector.process(args.num_procs)
    newlon = np.arange(projector.lonmin, projector.lonmax, 1./200.)
    newlat = np.arange(projector.latmin, projector.latmax, 1./200.)

    IMG, mask = map_project(newlon, newlat, "%s.nc"%projector.fname, save=True)
