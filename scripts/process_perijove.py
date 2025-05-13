from junocam_projection.projector import Projector
import numpy as np
import glob
import os
import json
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

parser = argparse.ArgumentParser(
    description="Process all images for a given perijoves to a cylindrical projection"
)
parser.add_argument(
    "--metadata_folder",
    help="Path to folder containing metadata files for the perijove",
    type=str,
    required=True,
)
parser.add_argument(
    "--image_folder",
    help="Path to the folder containing the JPG files for the perijove",
    type=str,
    required=True,
)
parser.add_argument(
    "--backplane_folder",
    help="Path to store the intermediate backplane information",
    type=str,
    required=True,
)
parser.add_argument(
    "--map_folder",
    help="Path to store the cylindrically projected maps",
    type=str,
    required=True,
)
parser.add_argument(
    "--map_resolution", help="Map resolution [pixels per degree]", type=int, default=25
)
parser.add_argument(
    "--num_processes",
    help="Number of processors to use for projection",
    type=int,
    default=1,
)
parser.add_argument(
    "--kernel_folder", help="Path to store SPICE kernels", type=str, default="./kernels"
)

args = parser.parse_args()

KERNEL_DATAFOLDER = "./kernels/"
files = sorted(glob.glob(os.path.join(args.metadata_folder, "*.json")))

if not os.path.exists(args.backplane_folder):
    os.makedirs(args.backplane_folder)

if not os.path.exists(args.map_folder):
    os.makedirs(args.map_folder)

for file in files:
    with open(file, "r") as infile:
        data = json.load(infile)
    fname = data["FILE_NAME"].replace("-raw.png", "")

    backplane_fname = os.path.join(args.backplane_folder, f"{fname}.nc")
    if os.path.exists(backplane_fname):
        logger.info(f"Found {os.path.basename(backplane_fname)}")
        proj = Projector.load(backplane_fname, args.kernel_folder)
        proj.apply_correction('ls')
    else:
        logger.info(f"Projecting {fname}")
        proj = Projector(args.image_folder, file, args.kernel_folder)
        proj.process(num_procs=args.num_processes, apply_correction="ls")
        proj.save(backplane_fname)

    pc_data = proj.project_to_cylindrical_fullglobe(resolution=args.map_resolution)

    np.save(
        os.path.join(args.map_folder, f"{fname}_map.npy"),
        pc_data.image / pc_data.image.max(),
    )
