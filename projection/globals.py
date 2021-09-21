import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import netCDF4 as nc
import numpy as np
from .cython_utils import *
import spiceypy as spice
import multiprocessing
import multiprocessing.sharedctypes as sct
from cartopy import crs as ccrs
from skimage import exposure, color, io
from skimage.segmentation import slic
from skimage.future import graph
from scipy.ndimage.filters import gaussian_filter


FRAME_HEIGHT = 128
FRAME_WIDTH  = 1648

## filter ids for B, G and R 
FILTERS     = ['B','G','R']
CAMERA_IDS  = [-61501, -61502, -61503]

NC_FOLDER  = './nc/'
MOS_FOLDER = './mos/'
RGB_FOLDER = './rgb/'
NPY_FOLDER = './npy/'
