import matplotlib
import os
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import numpy as np
from .cython_utils import *
import multiprocessing
import multiprocessing.sharedctypes as sct
from scipy.ndimage.filters import gaussian_filter
from cartopy import crs as ccrs
from skimage import exposure, color, io
from skimage.segmentation import slic
from skimage.future import graph
import spiceypy as spice
import netCDF4 as nc
FRAME_HEIGHT = 128
FRAME_WIDTH  = 1648

## filter ids for B, G and R 
FILTERS     = ['B','G','R']
CAMERA_IDS  = [-61501, -61502, -61503]

NC_FOLDER  = './nc/'
MOS_FOLDER = './mos/'
RGB_FOLDER = './rgb/'
NPY_FOLDER = './npy/'
