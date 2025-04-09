from junocam_projection.mosaic import blend_maps
import numpy as np
import glob
import os
import tqdm
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

parser = argparse.ArgumentParser(description="Process all images for a given perijoves to a cylindrical projection")
parser.add_argument("--map_folder", help="Path to the cylindrically projected maps (npy files)", type=str, required=True)
parser.add_argument("--sigma_filter", help="Filter width for smoothing", type=int, default=50)
parser.add_argument("--sigma_cut", help="Filter width for edge detection", type=int, default=50)
parser.add_argument("--output", help="Path to output mosaic (npy)", type=str, default="mosaic.npy")

args = parser.parse_args()

KERNEL_DATAFOLDER = './kernels/'
files = sorted(glob.glob(os.path.join(args.map_folder, '*.npy')))

logger.info(f"Found {len(files)} files")

maps = []

for file in tqdm.tqdm(files, desc='Loading files', dynamic_ncols=True):
    maps.append(np.load(file))

mosaic = blend_maps(np.asarray(maps), args.sigma_filter, args.sigma_cut)

np.save(args.output, mosaic)
