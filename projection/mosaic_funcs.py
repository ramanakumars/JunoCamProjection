import numpy as np
import gc
import os
import time
import matplotlib.pyplot as plt
import multiprocessing
import spiceypy as spice
import tqdm
from .globals import (
    NC_FOLDER,
    NPY_FOLDER,
    MASK_FOLDER,
    MOS_FOLDER,
    initializer,
)
from multiprocessing import sharedctypes as sct
from .cython_utils import image_mask_c
from .camera_funcs import FILTERS
import sys
import ctypes
from cartopy import crs as ccrs
from skimage import exposure, color
import netCDF4 as nc
from scipy.interpolate import griddata

NLAT_SLICE = 20
NLON_SLICE = 20

BOX_X = 50
BOX_Y = 50

shared_lat = None
shared_lon = None
shared_img = None
shared_LON = None
shared_LAT = None
shared_IMG = None
nlon = 0
nlat = 0
nfiles = 0

shared_IMGs = None
shared_Ls = None
shared_mask = None
ave_Ls = 0.0


def lonlat_to_thetaphi(lon, lat):
    theta = np.radians(90 - lat)

    phi = lon.copy()
    phi = 180 - phi
    phi = np.radians(phi)

    return theta, phi


def lonlat_to_xyz(lon, lat):
    theta = np.radians(lat)

    phi = lon.copy()
    phi = 180 - phi
    phi = np.radians(phi)

    x = np.zeros(len(phi))
    y = np.zeros(len(phi))
    z = np.zeros(len(phi))
    for i, (lon, lat) in enumerate(zip(phi, theta)):
        x[i], y[i], z[i] = spice.srfrec(599, lon, lat)

    return x, y, z


def map_project_multi(
    files,
    outfile="multi_proj_raw.nc",
    pixres=1.0 / 25.0,
    num_procs=1,
    extents=None,
    scorr_method="fft",
    load=False,
    **kwargs
):
    """
    Project multiple images into one mosaic. Use `scorr_method` to
    do lighting correction so that there are no visible seams.
    Each image in the list is map projected in the same resolution and then
    interpolated to match the mosaic grid.


    Parameters
    ----------
    files : list
        list of files from the projection pipeline to mosaic
    pixres : float
        pixel resolution in degrees/pixel
    num_procs : int
        number of processors to use (used in the projection and interpolation)
    extents : list (optional)
        bounding box in the format [lonmin, lonmax, latmin, latmax]
        (in degrees). Set this to None to let it automatically determine
        the extents from the extents of all the images
    scorr_method : string
        lighting correction method to use:
            - 'none': no correction
            - 'simple': lambertian correction (output = input/cos(incidence))
            - 'poly': fit a 5th order polynomial for the image brightness
                in (mu, mu0) space
            - 'fft': use an FFT to correct for brightness variations
                in the image
    load : bool
        Set to `True` to load data from files (generated from previous run).
        Set to `False` to regenerate the image.
    kwargs: arguments to pass to `map_project`
    """

    nfiles = len(files)

    lats = []
    lons = []
    incs = []

    # Load all the files and get the lat/lon
    # info as well as the image and lighting geometry
    for i, file in enumerate(files):
        with nc.Dataset(NC_FOLDER + file, "r") as dataset:
            lati = dataset.variables["lat"][:]
            loni = dataset.variables["lon"][:]
            inci = dataset.variables["incidence"][:]

            lon_rot = kwargs.get("lon_rot", 0.0)
            loni[loni != -1000] += lon_rot
            loni[(loni < -180.0) & (loni != -1000)] += 360.0
            loni[loni > 180.0] -= 360.0

            lats.append(lati)
            lons.append(loni)
            incs.append(inci)

    # mask out bad pixels
    masks = [
        (lats[i] != -1000)
        & (lons[i] != -1000)
        & (np.abs(incs[i]) < np.radians(89.0))
        for i in range(len(files))
    ]

    latmin = [lats[i][masks[i]].min() for i in range(len(files))]
    latmax = [lats[i][masks[i]].max() for i in range(len(files))]

    if extents is None:
        # determine the bounding box if the `extent` variable is not passed
        latmin = np.min([lats[i][masks[i]].min() for i in range(len(files))])
        latmax = np.max([lats[i][masks[i]].max() for i in range(len(files))])
        lonmin = np.min([lons[i][masks[i]].min() for i in range(len(files))])
        lonmax = np.max([lons[i][masks[i]].max() for i in range(len(files))])
    else:
        lonmin, lonmax, latmin, latmax = extents

    print(
        "Extents - lon: %.3f %.3f  lat: %.3f %.3f"
        % (lonmin, lonmax, latmin, latmax),
        flush=True,
    )
    lats = None
    lons = None
    incs = None

    # create the uniform grid for mosaicing
    newlon = np.arange(lonmin, lonmax, pixres)
    newlat = np.arange(latmin, latmax, pixres)

    # LAT, LON = np.meshgrid(newlat, newlon)

    nlat = newlat.size
    nlon = newlon.size

    # create the arrays to hold the intermediary data
    IMG = np.zeros((nlat, nlon, 3), dtype=float)
    IMGs = np.zeros((len(files), nlat, nlon, 3), dtype=np.float32)
    Ls = np.zeros((len(files), nlat, nlon), dtype=np.float32)
    INCDs = np.zeros((len(files), nlat, nlon), dtype=np.float32)
    EMISs = np.zeros((len(files), nlat, nlon), dtype=np.float32)

    print("Mosaic shape: %d x %d" % (nlon, nlat), flush=True)

    for i, file in enumerate(files):
        # generate the projection of each image in the mosaic
        fname = files[i][:-3]
        _, IMGi, _, INCDs[i, :], EMISs[i, :] = map_project(
            file,
            pixres=pixres,
            long=newlon,
            latg=newlat,
            save=True,
            savemask=False,
            num_procs=num_procs,
            interp_loaded_grid=True,
            scorr_method=scorr_method,
            load=load,
            ret_inc=True,
            **kwargs
        )

        IMGi[np.isnan(IMGi)] = 0.0
        # get brightness information
        IMGs[i, :] = IMGi.astype(np.float32)
        Ls[i, :] = color.rgb2hsv(IMGi)[:, :, 2]

        # flush out any progress bars
        sys.stdout.flush()

    del IMGi
    gc.collect()

    combine_method = kwargs.get("combine_method", "box_average")

    # experimental method (not recommended)
    if combine_method == "min_grad":
        Lx, Ly = np.gradient(Ls, axis=(1, 2))
        Lxx = np.gradient(Lx, axis=1)
        Lyy = np.gradient(Ly, axis=2)
        delsqL = np.abs(Lx + Ly)

        for n in range(delsqL.shape[0]):
            delsqL[n, :, :] = get_fft(delsqL[n, :, :], 500)

    # get the overlap area, and correct for
    # brightness variations between overlap
    # regions in different images
    npix = np.sum(Ls > 0.005, axis=0)
    overlap_mask = npix > 1

    # get the average value of the overlap
    # region in each image
    ave_val = np.zeros(len(files))
    for i in range(len(files)):
        imgi = Ls[i, :][overlap_mask]
        ave_val[i] = (imgi[imgi > 0.0]).mean()

    # get the average of the all overlaps
    ave_all = np.mean(ave_val)
    print(ave_val, ave_all)

    # correct each image so that the overlap
    # regions have the same brightness
    for i in range(len(files)):
        IMGs[i, :] *= ave_all / ave_val[i]
        Ls[i, :] = color.rgb2hsv(IMGs[i, :])[:, :, 2]

    # clear the memory
    del overlap_mask
    del ave_val
    del npix
    gc.collect()

    if combine_method not in ["none"]:
        # create a dummy lighting geometry variable
        incem = INCDs * EMISs
    else:
        incem = None

    # build the image mask for each input file first
    imgs_mask = np.zeros(IMGs.shape[:-1], dtype=np.uint8)
    for jj in tqdm.tqdm(range(IMG.shape[0]), desc="Building image mask"):
        # if jj%100==0:
        #    print("\rBuilding image mask: [%-20s] %d/%d"%(int(jj/IMG.shape[0]*20.)*'=', jj+1, IMG.shape[0]),
        #          end='', flush=True)
        for ii in range(IMG.shape[1]):
            incemij = incem[:, jj, ii]

            # the best pixel is one that actually saw that feature, is not too dim,
            # and does not have a low incidence angle
            imgs_mask[:, jj, ii] = (
                ~np.isnan(incemij)
                & (Ls[:, jj, ii] > 0.5 * Ls[:, jj, ii].max())
                & (INCDs[:, jj, ii] < np.radians(80))
            )

    sys.stdout.flush()

    # save these parameters to a NetCDF file so that we can plot it later
    with nc.Dataset(NC_FOLDER + outfile, "w") as f:
        xdim = f.createDimension("x", nlon)
        ydim = f.createDimension("y", nlat)
        colors = f.createDimension("colors", 3)
        files = f.createDimension("file", len(files))

        # create the NetCDF variables
        latVar = f.createVariable("lat", "float64", "y", zlib=True)
        lonVar = f.createVariable("lon", "float64", "x", zlib=True)
        imgsVar = f.createVariable(
            "imgs", "float32", ("file", "y", "x", "colors"), zlib=True
        )
        masksVar = f.createVariable(
            "img_mask", "int8", ("file", "y", "x"), zlib=True
        )

        latVar[:] = newlat[:]
        lonVar[:] = newlon[:]
        imgsVar[:] = IMGs[:]
        masksVar[:] = imgs_mask[:]

    if combine_method == "max":
        combine = np.max
    elif combine_method == "min":
        combine = np.min

    if "none" in scorr_method:
        IMG = np.max(IMGs, axis=0)
    else:
        # remove regions of low lighting (even after
        # lighting correction)
        incem[IMGs.sum(axis=-1) < 0.02] = np.nan
        print("Mosaicing image", flush=True)

        # loop through every pixel and find the best value to assign to the
        # mosaic
        if combine_method in ["min", "max"]:
            for jj in tqdm.tqdm(range(IMG.shape[0])):
                for ii in range(IMG.shape[1]):
                    incemij = incem[:, jj, ii]

                    # the best pixel is one that actually saw that feature, is not too dim,
                    # and does not have a low incidence angle
                    mask = (
                        ~np.isnan(incemij)
                        & (Ls[:, jj, ii] > 0.5 * Ls[:, jj, ii].max())
                        & (INCDs[:, jj, ii] < np.radians(80))
                    )
                    nimgs = np.sum(mask)
                    if nimgs > 1:
                        # set the value of the pixel in the mosaic
                        if combine_method == "min_grad":
                            IMG[jj, ii, :] = IMGs[
                                np.argmin(delsqL[mask, jj, ii]), jj, ii, :
                            ]
                        else:
                            IMG[jj, ii, :] = combine(
                                IMGs[mask, jj, ii, :], axis=0
                            )
                    elif nimgs > 0:
                        IMG[jj, ii, :] = IMGs[mask, jj, ii, :]
        else:
            # clean up memory usage
            IMGs = None
            INCDs = None
            Ls = None
            gc.collect()

            IMG = box_average(
                NC_FOLDER + outfile, ave_all, num_procs=num_procs
            )

    # save these parameters to a NetCDF file so that we can plot it later
    with nc.Dataset(NC_FOLDER + outfile, "r+") as f:
        if "img" not in f.variables.keys():
            imgVar = f.createVariable(
                "img", "float64", ("y", "x", "colors"), zlib=True
            )
            imgVar[:] = IMG[:]
        else:
            f.variables["img"] = IMG[:]

    # normalize the image by the 99% percentile
    IMG = IMG / (np.percentile(IMG[IMG > 0.0], 99.0))
    IMG = np.clip(IMG, 0.0, 1.0)

    plt.imsave(MOS_FOLDER + "mosaic_RGB.png", IMG, origin="lower")
    return (newlon, newlat, IMG)


def map_project(
    file,
    long=None,
    latg=None,
    pixres=None,
    num_procs=1,
    save=False,
    savemask=False,
    scorr_method="simple",
    load=False,
    ret_inc=False,
    **kwargs
):
    """
    Interpolate a single file onto a regular lon/lat grid. By default,
    the images are created in a grid that is 5 degrees bigger than the
    extent of the raw image observed by JunoCam. Then, this is interpolated
    to fit the required mesh. This is in an effort to segment the mosaicing
    process, where we can generate a set of map-projected images beforehand,
    and then do the mosaicing process afterwords, without wasting too much
    space with unnecessary data.

    Parameters
    ----------
    newlon : numpy.ndarray
        The new regularly intervaled longitude array
    newlat : numpy.ndarray
        The new regularly intervaled latitude array
    file : string
        Input netCDF file with the projected data
    num_procs : int
        Number of threads to use to grid [Default: 1]
    save : bool
        Use True to save the output grid data into a
        netCDF file [Default: False]
    savemask : bool
        Use True to save the mask data as a PNG file [Default: False]
    scorr_method : string
        The type of lighting correction to use:
        simple : assume Jupiter is a Lambertian surface (works decently
                well for some images but does not handle terminators well)
        poly :   fit a 2nd order polynomial in mu and mu0 based on the
                brightness of the image
        fft : do a FFT-based high pass filter to remove low-frequency (large scale)
                variations in the brightness of the image. Use the colorspace
                variable to pass the colorspace that is used to get the brightness
                (e.g., V in HSV space, L in Lab space, etc.)
        combination: you can combine several methods together. e.g., 'simple+fft'
                will run both the simple Lambertian and the FFT. (not recommended)
    load : bool
        Set to `True` to load data from files (generated from previous run).
        Set to `False` to regenerate the image.
    ret_inc : bool
        Set to `True` to return the incidence and emission angle data.
    **kwargs : arguments to pass to the scorr functions
    """
    from scipy.interpolate import RegularGridInterpolator

    global shared_LON, shared_LAT, nlon, nlat

    fname = file[:-3]
    print("Projecting %s" % fname)

    if not os.path.exists(MOS_FOLDER):
        os.mkdir(MOS_FOLDER)
    if not os.path.exists(MASK_FOLDER):
        os.mkdir(MASK_FOLDER)
    if not os.path.exists(NPY_FOLDER):
        os.mkdir(NPY_FOLDER)

    # open the file and load the data
    dataset = nc.Dataset(NC_FOLDER + file, "r")
    lats = dataset.variables["lat"][:]
    lons = dataset.variables["lon"][:]
    use_flux = kwargs.get("use_flux", False)
    if use_flux:
        imgs = dataset.variables["flux"][:].astype(float)
    else:
        print("Using raw IMG")
        imgs = dataset.variables["img"][:].astype(float)
    scloci = dataset.variables["scloc"][:]
    eti = dataset.variables["et"][:]
    incid = dataset.variables["incidence"][:]
    emis = dataset.variables["emission"][:]

    # rotate the lon grid if needed
    lon_rot = kwargs.get("lon_rot", 0.0)
    lons[lons != -1000] += lon_rot
    lons[(lons < -180.0) & (lons != -1000)] += 360.0
    lons[lons > 180.0] -= 360.0

    nframes = eti.shape[0]

    if pixres is not None:
        masks = (
            (lats != -1000)
            & (lons != -1000)
            & (np.abs(incid) < np.radians(89.0))
        )
        latmin = lats[masks].min()
        latmax = lats[masks].max()
        lonmin = lons[masks].min()
        lonmax = lons[masks].max()

        nx = int((lonmax - lonmin) / pixres + 1)
        ny = int((latmax - latmin) / pixres + 1)

        newlon = np.linspace(lonmin, lonmax, nx, endpoint=True)
        newlat = np.linspace(latmin, latmax, ny, endpoint=True)
        print(
            "Limits: lon: %.3f %.3f  lat: %.3f %.3f  size: %d x %d"
            % (
                newlon.min(),
                newlon.max(),
                newlat.min(),
                newlat.max(),
                newlon.size,
                newlat.size,
            )
        )
        print(newlat[-1] - latmax, newlon[-1] - lonmax)
    else:
        raise RuntimeError("Please provide a resolution")

    # define the arrays to hold the new gridded data
    nlat = newlat.size
    nlon = newlon.size
    IMG = np.zeros((nlat, nlon, 3))
    LAT, LON = np.meshgrid(newlat, newlon)

    # get the image mask where no data exists
    # this is created to remove errors from interpolation
    try:
        if load & os.path.exists(NPY_FOLDER + "%s_mask.npy" % fname):
            maski = np.load(NPY_FOLDER + "%s_mask.npy" % fname)
            if maski.shape != LON.T.shape:
                raise ValueError("mask has incorrect shape")
        else:
            raise ValueError("load is not enabled")

        print("Loaded mask file")
    except ValueError as e:
        print(e)
        roll_lon = newlon - lon_rot
        roll_lon[roll_lon < -180.0] += 360.0
        roll_lon[roll_lon > 180.0] -= 360.0
        output = image_mask_c(
            np.radians(newlat), np.radians(roll_lon), nlat, nlon, eti, nframes
        )
        maski = ctypes.cast(
            output, ctypes.POINTER(ctypes.c_int * (nlat * nlon))
        ).contents
        maski = np.asarray(maski, dtype=np.int).reshape((nlat, nlon))
        np.save(NPY_FOLDER + "%s_mask.npy" % fname, maski)

    # save the mask and the raw pixel values if needed
    if savemask:
        plt.imsave(
            MASK_FOLDER + "mask_%s.png" % (fname),
            maski,
            vmin=0.0,
            vmax=1.0,
            cmap="gray",
            origin="lower",
        )

    # create a shared memory object for LON/LAT
    LON_ctypes = np.ctypeslib.as_ctypes(LON)
    shared_LON = sct.RawArray(LON_ctypes._type_, LON_ctypes)
    LAT_ctypes = np.ctypeslib.as_ctypes(LAT)
    shared_LAT = sct.RawArray(LAT_ctypes._type_, LAT_ctypes)

    # loop through each color and create that color band in the projection
    for ci in range(3):
        print("Processing %s" % (FILTERS[ci]))
        filteriname = NPY_FOLDER + "%s_%s.npy" % (fname, FILTERS[ci])
        # load the image if needed
        try:
            if load & os.path.exists(filteriname):
                print("Loading from %s" % filteriname)
                IMGI = np.load(filteriname)
                maski[IMGI < 0.001] = 0
                IMG[:, :, ci] = IMGI
            else:
                raise ValueError("load is not enabled")
        except (ValueError, IndexError):
            lati = lats[:, ci, :, :].flatten()
            loni = lons[:, ci, :, :].flatten()
            imgi = imgs[:, ci, :, :].flatten()
            emi = emis[:, ci, :, :].flatten()
            inci = incid[:, ci, :, :].flatten()

            invmask = np.where(
                (lati == -1000.0)
                | (loni == -1000.0)
                | (np.abs(inci) > np.radians(85.0))
            )[0]
            # remove pixels that were not projected
            lat = np.delete(lati, invmask)
            lon = np.delete(loni, invmask)
            img = np.delete(imgi, invmask)

            # do the gridding
            IMGI = project_to_uniform_grid(lon, lat, img, num_procs)

            # remove interpolation errors
            IMGI[np.isnan(IMGI)] = 0.0
            IMGI[IMGI < 0.0] = 0.0

            # Save the data out to a numpy file
            np.save(filteriname, IMGI)
            maski[IMGI < 0.001] = 0
            IMG[:, :, ci] = IMGI

    INCD, EMIS = get_emis_incid_map(
        incid,
        emis,
        lats,
        lons,
        newlat,
        newlon,
        maski,
        fname,
        num_procs=num_procs,
        load=load,
    )

    maski = np.clip(maski, 0, 1)
    for ci in range(3):
        IMG[:, :, ci] = maski * IMG[:, :, ci]

    # switch from BGR to RGB
    IMG = IMG[:, :, ::-1]

    # Do a simple Lambertian correction if needed
    if "simple" in scorr_method:
        mu0 = np.clip(np.cos(INCD), 0, 1)
        mu0[mu0 < 0.01] = np.nan
        scorr = mu0  # 2.*mu0/(mu0 + mu)
        for ci in range(3):
            IMG[:, :, ci] = IMG[:, :, ci] / scorr

    if "poly" in scorr_method:
        IMG = scorr_poly(INCD, EMIS, newlat, newlon, IMG)

    # with both the poly and simple corrections,
    # bad incidence values are treated as NaNs.
    # remove those pixels here
    IMG[np.isnan(IMG)] = 0.0

    if "fft" in scorr_method:
        # do an FFT correction if needed
        IMG = scorr_fft(
            IMG,
            fname,
            radius=kwargs.get("fft_radius", 5.0),
            colorspace=kwargs.get("colorspace", "hsv"),
            trim_rad=kwargs.get("trim_rad", 0.6),
            trim_threshold=kwargs.get("trim_thresh", 0.95),
        )

    # expand the image out to the required grid
    interp_loaded_grid = kwargs.get("interp_loaded_grid", False)
    if interp_loaded_grid:
        assert (latg is not None) & (
            long is not None
        ), "Please provide the grid lat/lon"
        IMGnew = np.zeros((latg.size, long.size, 3))
        LATG, LONG = np.meshgrid(latg, long)
        newpoints = np.stack((LATG.flatten(), LONG.flatten()), axis=1).reshape(
            -1, 2
        )
        print(
            "Interpolating from %d x %d => %d x %d"
            % (nlat, nlon, latg.size, long.size)
        )
        for ci in range(3):
            interp_function = RegularGridInterpolator(
                (newlat, newlon),
                IMG[:, :, ci],
                bounds_error=False,
                fill_value=0.0,
            )
            imgi = interp_function(newpoints).reshape((long.size, latg.size)).T
            IMGnew[:, :, ci] = imgi

        # remove interpolation errors where one color has dark data but the
        # other don't. these needs to be large enough to ignore dark areas,
        # but small enough that we don't miss color features (such as GRS)
        IMGnew = IMGnew.reshape((long.size * latg.size, 3))
        IMGnew[IMGnew.min(axis=-1) < 0.1, :] = 0
        IMG = IMGnew.reshape((latg.size, long.size, 3))
        imgi = IMG.copy()

        # get the edge points by plotting a contour around the image
        # cs = plt.contour(np.arange(imgi.shape[1]), np.arange(imgi.shape[0]), imgi.mean(axis=-1), [0, 0.1])
        # edge_points = measure.find_contours(imgi.mean(axis=-1), 0.1)#cs.collections[0].get_paths()[0].vertices.astype(int)
        # edge_points = np.asarray(edge_points[0], dtype=int)

        trim_size = 2

        """
        # loop over the edge points and trim
        for i in range(len(edge_points)):
            if os.environ.get('NO_VERBOSE') is None:
                print("\rTrimming: [%-20s] %d/%d"%(int(i/len(edge_points)*20)*'=', i+1, len(edge_points)), end='')
            jj, ii = edge_points[i,:]
            # get the window around this pixel
            img_sub = IMG[(jj-trim_size):(jj+trim_size),(ii-trim_size):(ii+trim_size),:]

            # if there is an edge in this image (given by a 0 value)
            # set the pixel value to 0
            if img_sub.mean(axis=-1).min() < 0.1:
                imgi[jj,ii,:] = 0.
        """
        trim_size = 2
        # trim edges from this image to avoid interpolation errors
        for jj in tqdm.tqdm(
            range(trim_size, latg.size - trim_size), desc="Trimming"
        ):
            # ignore if this column has no data (speeds up trim computation)
            if IMG[jj, :, :].max() == 0:
                continue
            for ii in range(trim_size, long.size - trim_size):
                if IMG[jj, ii, :].mean() == 0:
                    continue
                # get the window around this pixel
                img_sub = IMG[
                    (jj - trim_size) : (jj + trim_size),
                    (ii - trim_size) : (ii + trim_size),
                    :,
                ]

                # if there is an edge in this image (given by a 0 value)
                # set the pixel value to 0
                if img_sub.mean(axis=-1).min() < 0.1:
                    imgi[jj, ii, :] = 0.0
        print()

        IMG = imgi.copy()

        # interpolate the other variables
        interp_function = RegularGridInterpolator(
            (newlat, newlon), INCD, bounds_error=False, fill_value=np.nan
        )
        INCD = interp_function(newpoints).reshape((long.size, latg.size)).T
        interp_function = RegularGridInterpolator(
            (newlat, newlon), EMIS, bounds_error=False, fill_value=np.nan
        )
        EMIS = interp_function(newpoints).reshape((long.size, latg.size)).T
        interp_function = RegularGridInterpolator(
            (newlat, newlon), maski, bounds_error=False, fill_value=0.0
        )
        maski = interp_function(newpoints).reshape((long.size, latg.size)).T

        # update the lat/lon info for output
        newlat = latg
        newlon = long
        nlon = long.size
        nlat = latg.size

    IMGc = IMG / IMG.max()
    IMGc = np.clip(IMGc, 0.0, 1.0)  # IMG/np.percentile(IMG, 99)

    if save:
        plt.imsave(
            MOS_FOLDER + "%s_mosaic.png" % (fname), IMGc, origin="lower"
        )

    # save these parameters to a NetCDF file so that we can plot it later
    with nc.Dataset(NC_FOLDER + "%s_proj.nc" % fname, "w") as f:
        xdim = f.createDimension("x", nlon)
        ydim = f.createDimension("y", nlat)
        colors = f.createDimension("colors", 3)

        # create the NetCDF variables
        latVar = f.createVariable("lat", "float64", "y")
        lonVar = f.createVariable("lon", "float64", "x")

        lonVar.units = "degrees east"
        latVar.units = "degrees north"

        imgVar = f.createVariable(
            "img", "float64", ("y", "x", "colors"), zlib=True
        )

        img_corrVar = f.createVariable(
            "img_corr", "uint8", ("y", "x", "colors"), zlib=True
        )

        incdVar = f.createVariable("incd", "float32", ("y", "x"), zlib=True)
        emisVar = f.createVariable("emis", "float32", ("y", "x"), zlib=True)

        latVar[:] = newlat[:]
        lonVar[:] = newlon[:]
        imgVar[:] = IMG[:]
        incdVar[:] = INCD[:]
        emisVar[:] = EMIS[:]
        img_corrVar[:] = np.asarray(IMGc * 255 / IMG.max(), dtype=np.uint8)

    if ret_inc:
        return "%s_proj.nc" % fname, IMG, maski, INCD, EMIS
    else:
        return "%s_proj.nc" % fname, IMG, maski


def project_to_uniform_grid(lon, lat, img, num_procs=1):
    """
    Main driver for the regridding. Handles the multiprocessing
    part of the process.

    Parameters
    ----------
    lon : numpy.ndarray
        Original unstructured longitude data
    lat : numpy.ndarray
        Original unstructured latitude data
    img : numpy.ndarray
        Unstructured image data corresponding to that lon/lat
    num_procs : int
        number of threads to create to grid the data

    Returns
    -------
    IMG : numpy.ndarray
        Data interpolated onto a regular grid

    """
    global shared_img, shared_lon, shared_lat, shared_LON, shared_LAT, shared_IMG, nlon, nlat
    nsquare_lon = int(np.ceil(nlon / NLON_SLICE))
    nsquare_lat = int(np.ceil(nlat / NLAT_SLICE))

    # conver the data arrays into shared memory
    """
    lon_ctypes = np.ctypeslib.as_ctypes(lon)
    shared_lon = sct.RawArray(lon_ctypes._type_, lon_ctypes)
    lat_ctypes = np.ctypeslib.as_ctypes(lat)
    shared_lat = sct.RawArray(lat_ctypes._type_, lat_ctypes)
    img_ctypes = np.ctypeslib.as_ctypes(img)
    shared_img = sct.RawArray(img_ctypes._type_, img_ctypes)
    """
    inpargs = []
    indices = []

    # convert back to a numpy array to process
    LON = np.asarray(shared_LON, dtype=np.float32).reshape(nlon, nlat)
    LAT = np.asarray(shared_LAT, dtype=np.float32).reshape(nlon, nlat)

    # get the pixel resolution in deg/pixel
    # used as a search radius to get the points within the
    # domain bounds
    pixres = LON[1, 0] - LON[0, 0]

    # build the inputs to the multiprocessing pipeline
    # this will decompose the longitude grid into NLON_SLICE
    # and the latitude grid into NLAT_SLICEs
    for j in range(NLAT_SLICE):
        startyind = j * nsquare_lat
        endyind = min([nlat, (j + 1) * nsquare_lat])
        for i in range(NLON_SLICE):
            startxind = i * nsquare_lon
            endxind = min([nlon, (i + 1) * nsquare_lon])
            LONi = LON[startxind:endxind, startyind:endyind]
            LATi = LAT[startxind:endxind, startyind:endyind]

            lonmin = LONi.min()
            lonmax = LONi.max()
            latmin = LATi.min()
            latmax = LATi.max()

            maski = np.where(
                (lon > lonmin - 50 * pixres)
                & (lon < lonmax + 50 * pixres)
                & (lat > latmin - 50 * pixres)
                & (lat < latmax + 50 * pixres)
            )[0]
            # make sure there is enough data to grid
            if len(maski) > 3:
                # inpargs.append([startxind,endxind,startyind,endyind, maski])
                inpargs.append(
                    [
                        startxind,
                        endxind,
                        startyind,
                        endyind,
                        lon[maski],
                        lat[maski],
                        img[maski],
                    ]
                )
    # create the final image array
    # this will be stored as a shared array so each process
    # can write to it
    cdtype = np.ctypeslib.as_ctypes_type(np.dtype(float))
    shared_IMG = multiprocessing.RawArray(cdtype, nlat * nlon)

    # start the pool
    pool = multiprocessing.Pool(processes=num_procs, initializer=initializer)
    try:
        i = 0

        # start the multicore grid processing
        r = pool.map_async(project_part_image, inpargs, chunksize=5)
        pool.close()

        tasks = pool._cache[r._job]
        ninpt = len(inpargs)
        if os.environ.get("NO_VERBOSE") is None:
            with tqdm.tqdm(total=ninpt) as pbar:
                while tasks._number_left > 0:
                    # progress = (ninpt - tasks._number_left*tasks._chunksize)/ninpt
                    pbar.n = max(
                        [0, ninpt - tasks._number_left * tasks._chunksize]
                    )
                    pbar.refresh()

                    #    print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
                    time.sleep(0.1)
    except Exception as e:
        print(e)
        pool.terminate()
        pool.join()
        sys.exit()

    pool.join()

    # np.ctypeslib.as_array(IMG_ctypes)
    return np.frombuffer(shared_IMG, dtype=float).reshape((nlat, nlon))


def project_part_image(inp):
    """
    Main gridding code. Interpolates the unstructured data onto
    a regular grid

    Parameters
    ----------
    inp : tuple
        Contains startxind, endxind, startyind, endyind and mask
        The indices define the slice of the new image array that will be
        processed here while mask is the index of the original unstructured
        coordinates that are within this domain.

    Outputs
    -------
    imgi : numpy.ndarray
        The interpolated array on a regular grid corresponding to
        [startyind:endyind,startxind:endxind] of the full image
    """
    global shared_lon, shared_lat, shared_img, shared_LON, shared_LAT, shared_IMG, nlon, nlat

    # startxind,endxind,startyind,endyind, maski = inp
    startxind, endxind, startyind, endyind, lon, lat, img = inp

    # lon = np.asarray(shared_lon, dtype=np.float32)[maski]
    # lat = np.asarray(shared_lat, dtype=np.float32)[maski]
    # img = np.asarray(shared_img, dtype=np.float64)[maski]
    LON = np.asarray(shared_LON, dtype=np.float32).reshape(nlon, nlat)[
        startxind:endxind, startyind:endyind
    ]
    LAT = np.asarray(shared_LAT, dtype=np.float32).reshape(nlon, nlat)[
        startxind:endxind, startyind:endyind
    ]

    IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat, nlon))

    x, y, z = lonlat_to_xyz(lon, lat)
    X, Y, Z = lonlat_to_xyz(LON.T.ravel(), LAT.T.ravel())

    points = np.dstack([x, y, z])[0, :]
    newpoints = np.dstack([X, Y, Z])[0, :]

    # points    = np.dstack([lon, lat])[0,:]
    # newpoints = np.dstack([LON.T.ravel(), LAT.T.ravel()])[0,:]

    try:
        img[np.isnan(img)] = 0.0
        imgi = griddata(
            points, img, newpoints, method="nearest", fill_value=0.0
        ).reshape(LON.T.shape)
        # imgi =  griddata((lon, lat), \
        #                 img, (LON, LAT), method='nearest').T
        IMG[startyind:endyind, startxind:endxind] = imgi
        # return (startxind, endxind, startyind, endyind, imgi)
    except Exception as e:
        print(e)
        raise e


def color_correction(
    datafile,
    gamma=1.0,
    hist_eq=True,
    fname=None,
    save=False,
    trim_saturated=False,
    sat_threshold=95,
    clip_limit=0.008,
    **kwargs
):
    """
    Do the color and gamma correction, and image scaling on the image

    Parameters
    ----------
    IMG : numpy.ndarray
        Input image
    newlon : numpy.ndarray
        Regularly intervalled longitude array
    newlat : numpy.ndarray
        Regularly intervalled latitude array
    gamma : float
        Gamma correction -- img_out = img**gamma [Default : 1.0]
    fname : string
        filename to save the image [Default: None] -- see `save`
    save : bool
        If true, save to output PNG defined as `fname_mosaic_RGB.png`
    hist_eq : bool
        Toggle `True` to do histogram equalization to enhance
    clip_limit : float
        Threshold for histogram equalization.
        See https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist

    Outputs
    -------
    IMG2 : numpy.ndarray
        Output color and gamma corrected file with 99-percentile scaling

    Raises
    ------
    AssertionError
        if `save`=True but no fname defined
    """
    with nc.Dataset(datafile, "r") as data:
        IMG = data.variables["img"][:]
        lon = data.variables["lon"][:]
        lat = data.variables["lat"][:]

        IMG2 = IMG.copy()

        if trim_saturated:
            scaling = np.percentile(IMG2[IMG2 > 0.0], sat_threshold)
            IMGc = IMG2 / scaling
            IMGc[IMGc.sum(axis=2) > 2.98] = 0.0
            IMG2 = IMGc * scaling

        if hist_eq:
            # normalize the image by the 95% percentile
            IMG2 = IMG2 / (np.percentile(IMG2[IMG2 > 0.0], 99.9))
            IMG2 = np.clip(IMG2, 0, 1)
            for ci in range(3):
                IMG2[:, :, ci] = exposure.equalize_adapthist(
                    IMG2[:, :, ci], clip_limit=clip_limit
                )

        IMG2 = IMG2**gamma
        # normalize the image by the 99.9% percentile
        IMG2 = IMG2 / (np.percentile(IMG2[IMG2 > 0.0], 99.9))
        IMG2 = np.clip(IMG2, 0, 1)

        IMG_corr = np.asarray(IMG2 * 255, dtype=np.uint8)

    with nc.Dataset(datafile, "r+") as data:
        # save the new image out to the netCDF file
        if "img_corr" not in data.variables.keys():
            img_corrVar = data.createVariable(
                "img_corr", "uint8", ("y", "x", "colors"), zlib=True
            )
            img_corrVar[:] = IMG_corr
        else:
            data.variables["img_corr"][:] = IMG_corr

    if save:
        assert not isinstance(
            fname, type(None)
        ), "please provide a filename to save the data"
        plt.imsave(
            MOS_FOLDER + "%s_mosaic_RGB.png" % fname, IMG2, origin="lower"
        )

    """
    fig, ax = plt.subplots(1,1,figsize=(10,10),dpi=150)
    ax.imshow(IMG2, extent=(lon.min(), lon.max(), lat.min(), lat.max()),\
               origin='lower')
    ax.set_xlabel(r"Longitude [deg]")
    ax.set_ylabel(r"Latitude [deg]")
    plt.show()
    """

    return IMG2


def plot_ortho(datafile, sat_height_scale=1.0, facecolor="black"):
    with nc.Dataset(datafile, "r") as data:
        if "img_corr" in data.variables.keys():
            IMG = data.variables["img_corr"][:]
        else:
            IMG = data.variables["img"][:]
        lon = data.variables["lon"][:]
        lat = data.variables["lat"][:]

    LON, LAT = np.meshgrid(lon, lat)

    proj = ccrs.NearsidePerspective(
        central_longitude=LON[IMG.min(axis=2) > 0.0].mean(),
        central_latitude=LAT[IMG.min(axis=2) > 0.0].mean(),
        satellite_height=35785.0e3 * sat_height_scale,
    )

    fig = plt.figure(figsize=(10, 10), dpi=150, facecolor=facecolor)
    ax = fig.add_subplot(projection=proj)

    ax.imshow(
        IMG[::-1, :],
        origin="upper",
        extent=(lon.min(), lon.max(), lat.min(), lat.max()),
        transform=ccrs.PlateCarree(),
        interpolation="hermite",
    )

    plt.show()


def get_emis_incid_map(
    incid, emis, lat, lon, newlat, newlon, maski, fname, num_procs=1, load=True
):
    # flatten and mask out the bad data points
    incid = incid.flatten()
    emis = emis.flatten()
    lat = lat.flatten()
    lon = lon.flatten()
    mask = (lat != -1000) & (lon != -1000) & (np.abs(incid) < np.radians(89.0))

    incidsf = incid[mask].flatten()
    emisf = emis[mask].flatten()
    lonf = lon[mask].flatten()
    latf = lat[mask].flatten()

    # create the new grid to project onto and
    # project the incidence and emission values onto the new grid
    LAT, LON = np.meshgrid(newlat, newlon)

    try:
        if load & os.path.exists(NPY_FOLDER + "%s_emis.npy" % fname):
            print("Loading emission data")
            EMIS = np.load(NPY_FOLDER + "%s_emis.npy" % fname)
        else:
            raise ValueError("load is not set.")

        if EMIS.shape != (newlat.size, newlon.size):
            raise ValueError("emis has incorrect size")
    except ValueError as e:
        print(e)
        print("Processing emission angles")
        EMIS = project_to_uniform_grid(lonf, latf, emisf, num_procs=num_procs)
        EMIS[~maski] = np.nan
        np.save(NPY_FOLDER + "%s_emis.npy" % fname, EMIS)

    try:
        if load & os.path.exists(NPY_FOLDER + "%s_incid.npy" % fname):
            print("Loading incidence data")
            INCD = np.load(NPY_FOLDER + "%s_incid.npy" % fname)
        else:
            raise ValueError("load is not set")

        if INCD.shape != (newlat.size, newlon.size):
            raise ValueError("emis has incorrect size")
    except ValueError as e:
        print(e)
        print("Processing incidence angles")
        INCD = project_to_uniform_grid(
            lonf, latf, incidsf, num_procs=num_procs
        )
        INCD[~maski] = np.nan
        np.save(NPY_FOLDER + "%s_incid.npy" % fname, INCD)

    return (INCD, EMIS)


def scorr_poly(INCD, EMIS, newlat, newlon, IMG):
    """
    Fit a 2nd order polynomial in incidence/emission space
    to correct for lighting geometry
    """
    print("Doing polynomial lighting correction")

    mask = (~np.isnan(EMIS)) & (~np.isnan(INCD)) & (IMG.min(axis=2) > 0.0)

    # get the brightness value of the pixels
    HSV = color.rgb2hsv(IMG / IMG.max())
    IMG_val = HSV[:, :, 2]
    print(IMG_val.min(), IMG_val.max())
    # IMG_val = IMG[:,:,1]

    INCDf = np.cos(INCD[mask].flatten())
    EMISf = np.cos(EMIS[mask].flatten())
    VALf = IMG_val[mask].flatten()

    # remove boundary data
    mask = INCDf > 0.05
    INCDf = INCDf[mask]
    EMISf = EMISf[mask]
    VALf = VALf[mask]

    ind = np.asarray(range(len(INCDf)))
    np.random.shuffle(ind)
    ind = ind[:1000]

    INCDf = INCDf[ind]
    EMISf = EMISf[ind]
    VALf = VALf[ind]

    # create the two axis (x is mu, y is mu0)
    x = INCDf
    y = EMISf
    xx = np.cos(INCD)
    yy = np.cos(EMIS)

    m = 5  # polynomial order
    A = []
    for i in range(m):
        for j in range(i):
            A.append((x**i) * (y**j))
    A = np.asarray(A).T

    # fit the polynomial
    coeff, r, rank, s = np.linalg.lstsq(A, VALf, rcond=None)

    # create the correction from the new polynomial
    SCORR = np.zeros_like(INCD)
    k = 0
    for i in range(m):
        for j in range(i):
            SCORR += coeff[k] * (xx**i) * (yy**j)
            k += 1

    # ignore negative values
    SCORR[SCORR < 0.0] = np.nan

    # correct the image and normalize
    HSV[:, :, 2] = HSV[:, :, 2] / SCORR
    HSV[:, :, 2] = HSV[:, :, 2] / np.percentile(
        HSV[:, :, 2][HSV[:, :, 2] > 0.0], 99
    )
    HSV[:, :, 2] = np.clip(HSV[:, :, 2], 0, 1)

    print(HSV[:, :, 2].min(), HSV[:, :, 2].max())

    # convert back to RGB
    IMG_corr = color.hsv2rgb(HSV)

    return IMG_corr


def scorr_fft(
    IMG, fname, radius=4.0, colorspace="hsv", trim_rad=0.7, trim_threshold=0.95
):
    """
    Do a correction using a FFT based high pass filter. This works by
    creating a mask of low-frequency (large scale) light variation which is used to divide
    the original image to remove light gradients in the image.

    Inputs
    ------
    IMG : numpy.ndarray
        the input image to be processed
    fname : string
        the name of the file (used for outputing the mask and FFT image)
    radius : float
        the radius of the Guassian filter that will be convolved
        with the image to extract the low frqeuency features
    trim_rad : float
        the radius to use to clip the image (to remove noise at the edges).
        The radius is given by `trim_rad*radius`, so `trim_rad` is the fraction/multiple
        of the original radius to use to trim the edges
    trim_threshold : float
        The threshold value for the trimming mask
        to remove edge noise. Values for the mask within the image will be 1
        while outside the image will be 0. Near the edges, the values will be
        in between. Choose a value close to 1 to clip close to the image, values
        close to zero will extend the edge.
    """
    from scipy.ndimage import center_of_mass

    scale = IMG.max()

    if colorspace == "hsv":
        data = color.rgb2hsv(IMG / scale)
        axis = 2
        invfunc = color.hsv2rgb
    elif colorspace == "lab":
        data = color.rgb2lab(IMG / scale).astype(float)
        axis = 0
        invfunc = color.lab2rgb
    elif colorspace == "yuv":
        data = color.rgb2yuv(IMG / scale)
        axis = 0
        invfunc = color.yuv2rgb
    elif colorspace == "rgb":
        data = IMG / IMG.max()
        axis = 1

        def invfunc(img):
            return img

    value = data[:, :, axis]

    # Center the image (better for the FFT)
    com = center_of_mass(value)
    dshift = (int(IMG.shape[0] // 2 - com[0]), int(IMG.shape[1] // 2 - com[1]))

    data = np.roll(data, dshift, axis=(0, 1))
    value = data[:, :, axis]

    # Do the FFT and get the filter
    ifft2 = get_fft(value, radius=radius)
    # plt.imsave(MASK_FOLDER+fname+"ifft.png", ifft2)

    # Divide the image by the filter to remove high frequency noise
    if colorspace != "rgb":
        valnew = value / ifft2
        data[:, :, axis] = valnew
    else:
        for ci in range(3):
            data[:, :, ci] = data[:, :, ci] / ifft2

        data = data / data.max()

    data = np.roll(data, (-dshift[0], -dshift[1]), axis=(0, 1))

    # Create a mask to trim the edges
    picmask = np.zeros_like(data[:, :, 0])

    # Find the pixels which contain image data
    picmask[invfunc(data).min(axis=-1) > 0.3] = 1.0

    # Filter the mask to blur the edges
    Lfilt = get_fft(picmask, radius=trim_rad * radius).flatten()

    # Trim the edge values based on the given threshold
    mask = Lfilt < trim_threshold

    # Save the mask as an image
    maskimg = np.zeros((data.shape[0] * data.shape[1]))
    maskimg[mask] = 1
    # plt.imsave(MASK_FOLDER+"%s_Lmask.png"%fname, maskimg.reshape((IMG.shape[0], IMG.shape[1])))

    # Trim the input image with this mask
    IMGf = data.reshape((data.shape[0] * data.shape[1], 3))
    IMGf[mask, :] = 0.0
    datanew = IMGf.reshape(IMG.shape)

    # Obtain the new image and transform it back to the original axis
    IMG = invfunc(datanew) * scale
    return IMG


def get_fft(value, radius):
    """
    Performs a FFT to convolve a Gaussian filter
    with the input image.

    Inputs
    ------
    value : numpy.ndarray
        The input 2D image to process
    radius : float
        The radius of the Gaussian filter (in pixels)

    Outputs
    -------
    ifft2 : numpy.ndarray
        The filter corresponding to the convolution
    """
    from scipy import fftpack

    # create the low pass filter
    xx = np.asarray(range(value.shape[1]))
    yy = np.asarray(range(value.shape[0]))
    XX, YY = np.meshgrid(xx, yy)

    dist = (XX - xx.mean()) ** 2.0 + (YY - yy.mean()) ** 2.0
    # exponential profile
    lp_filter = np.exp(-dist / (radius**2.0))

    # apply FFT to the value component
    f1 = fftpack.fftshift(fftpack.fft2(value))
    # f1  = fftpack.fftshift(fftpack.fft2(IMGr[:,:,ci]))

    filt = np.multiply(f1, lp_filter)

    # transform back to coordinate space
    ifft1 = np.real(fftpack.ifft2(fftpack.ifftshift(filt)))

    # limit the filter so we don't have unwanted growth
    ifft2 = np.clip(ifft1, 0.05, 1)

    return ifft2


def initialize_box_average(IMGs, Ls, mask):
    global shared_IMGs, shared_Ls, shared_mask

    shared_IMGs = IMGs
    shared_Ls = Ls
    shared_mask = mask

    initializer()


def box_average(file, ave_all=1000, num_procs=1):
    """
    Driver for the box_average stacking method. Calls the main function
    via a multiprocessing Pool.

    The method works by sliding a box (BOX_X, BOX_Y) in shape across the
    image and finding the mean brightness of each image in that box and
    applying a brightness weight such that the final image has a constant
    brightness across the entire domain. The final stacked pixel values
    within that box is a weighted pixel-wise mean of all images.


    Inputs
    ------
    IMGs : numpy.ndarray
        Array of images to stacked. Size is (nfiles, nlat, nlon, 3)
    INCDs : numpy.ndarray
        incidence angles of all images. Size is (nfiles, nlat, nlon)
    incem : numpy.ndarray
        product of cos(incidence) and cos(emission). Dummy lighting
        geometry variable used to mask bad pixels.
    Ls : numpy.ndarray
        array of brightness values of the images. Taken as the value
        channel of the HSV version of IMGs
    ave_all : float
        average brightness across all images
    num_procs : int
        number of processes to spawn for multiprocessing [default: 1]

    Outputs
    -------
    IMG : numpy.ndarray
        final stacked image. Size is (nlat, nlon, 3)
    """
    global shared_IMGs, shared_Ls, shared_mask, shared_IMG, nfiles, nlon, nlat, ave_Ls

    # open the file and load the variables
    with nc.Dataset(file, "r") as dset:
        nfiles = dset.dimensions["file"].size
        nlon = dset.dimensions["x"].size
        nlat = dset.dimensions["y"].size
        IMGs = dset.variables["imgs"][:]
        mask = dset.variables["img_mask"][:]

    print(IMGs.shape)
    # calculate the luminance
    Ls = IMGs.mean(axis=-1)  # color.rgb2hsv(IMGs[i,:])[:,:,2]

    print(Ls.min(), Ls.max())

    nx = int(np.ceil(nlon / BOX_X))
    ny = int(np.ceil(nlat / BOX_Y))

    inpargs = []

    ave_Ls = ave_all

    # build the inputs to the multiprocessing pipeline
    # this will decompose the imgs into (BOX_Y, BOX_X) slices
    for jj in range(ny):
        starty = jj * BOX_Y
        endy = min([nlat, (jj + 1) * BOX_Y])
        for ii in range(nx):
            startx = ii * BOX_X
            endx = min([nlon, (ii + 1) * BOX_X])

            # inpargs.append([startxind,endxind,startyind,endyind, maski])
            inpargs.append([startx, endx, starty, endy])
    print(len(inpargs))

    # create the final image array
    # this will be stored as a shared array so each process
    # can write to it
    t0 = time.process_time()

    # since this is all on *nix machines, we can take
    # advantage of the copy-on-write functionality, so just
    # share the array as a global variable, and this will
    # be accessible by all processes. we won't use additional
    # memory since these variables are only copied when they are
    # modified
    IMGs_ctypes = np.ctypeslib.as_ctypes(IMGs)
    shared_IMGs = np.ctypeslib.as_array(
        sct.RawArray(IMGs_ctypes._type_, IMGs_ctypes)
    ).reshape((nfiles, nlat, nlon, 3))
    Ls_ctypes = np.ctypeslib.as_ctypes(Ls)
    shared_Ls = np.ctypeslib.as_array(
        sct.RawArray(Ls_ctypes._type_, Ls_ctypes)
    ).reshape((nfiles, nlat, nlon))
    mask_ctypes = np.ctypeslib.as_ctypes(mask)
    shared_mask = np.ctypeslib.as_array(
        sct.RawArray(mask_ctypes._type_, mask_ctypes)
    ).reshape((nfiles, nlat, nlon))

    # the final image will need to be written to, so create a shared
    # multiprocessing object
    cdtype = np.ctypeslib.as_ctypes_type(np.dtype(float))
    shared_IMG = np.ctypeslib.as_array(
        sct.RawArray(cdtype, nlat * nlon * 3)
    ).reshape((nlat, nlon, 3))
    # shared_IMG      = np.ctypeslib.as_array(shared_IMG_base.get_obj()).reshape((nlat, nlon, 3))
    print(time.process_time() - t0, flush=True)

    # start the pool when memory is unloaded
    pool = multiprocessing.Pool(processes=num_procs, initializer=initializer)
    try:
        # start the multicore grid processing
        r = pool.map_async(do_average_box, inpargs)
        pool.close()

        tasks = pool._cache[r._job]
        ninpt = len(inpargs)
        last_update = tasks._number_left
        print("starting...")
        while tasks._number_left > 0:
            progress = (ninpt - tasks._number_left * tasks._chunksize) / ninpt
            # print a progress when the number of tasks has been updated
            if progress != last_update:
                print(
                    "\r[%-20s] %.2f%%"
                    % (int(progress * 20) * "=", progress * 100.0),
                    end="",
                )
                sys.stdout.flush()
                last_update = progress
            time.sleep(0.05)
        """
        for _ in tqdm.tqdm(pool.imap_unordered(do_average_box, inpargs),
                           total=len(inpargs), miniters=100,
                           mininterval=2):
            pass
        pool.close()
        """
    except Exception as e:
        pool.terminate()
        pool.join()
        raise e
        sys.exit()

    # pull the image array from the shared memory buffer
    # IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat,nlon, 3))
    # IMG = np.ctypeslib.as_array(shared_IMG_base.get_obj()).reshape((nlat,nlon, 3))
    # cleanup
    IMG = shared_IMG
    IMG[np.isnan(IMG)] = 0.0

    return IMG


def do_average_box(inp):
    """
    Main averaging code. Called by `box_average`. Does the calculation
    for a given box.

    Inputs
    -------
    inp : tuple
        grid extents for the given box. Values are the start and end pixel x
        coordinates and start and end pixel y coordinates for the box
    """
    global shared_IMGs, shared_Ls, shared_mask, shared_IMG, ave_Ls

    startx, endx, starty, endy = inp

    """
    IMGs   = np.array(shared_IMGs, dtype=np.float32).reshape((nfiles, nlat, nlon, 3))
    Ls     = np.array(shared_Ls, dtype=np.float32).reshape((nfiles, nlat, nlon))
    mask   = np.array(shared_mask, dtype=np.int8).reshape((nfiles, nlat, nlon))
    """

    ave_all = ave_Ls
    nfiles = shared_IMGs.shape[0]

    try:
        # get the images in this box
        mask_ij = shared_mask[:, starty:endy, startx:endx]

        if np.sum(mask_ij) > 1:
            Ls_ij = shared_Ls[:, starty:endy, startx:endx].astype(float)
            imgs_ij = shared_IMGs[:, starty:endy, startx:endx].astype(float)
            alpha = np.zeros_like(Ls_ij)

            # if there are, then assign weights (alpha) to each pixel in
            # each image. final image is a linear combination of images
            # with alphas
            for kk in range(nfiles):
                mask_k = np.where(mask_ij[kk, :] == 1)
                # weight each image by its relative brightness wrt to the
                # global mean
                if len(Ls_ij[kk, :, :][mask_k]) > 0:
                    alpha[kk, :, :][mask_k] = (
                        Ls_ij[kk, :, :][mask_k].mean() / ave_all
                    )
                    alpha[kk, :][alpha[kk, :] != 0] = (
                        1.0 / alpha[kk, :][alpha[kk, :] != 0.0]
                    )

            # IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat, nlon, 3))
            for c in range(3):
                IMGs_sub = imgs_ij[:, :, :, c] * alpha
                alpha_sum = np.sum(alpha, axis=0)
                alpha_sum[alpha_sum == 0] = np.nan
                shared_IMG[starty:endy, startx:endx, c] = np.divide(
                    np.sum(IMGs_sub, axis=0), alpha_sum
                )
    except Exception as e:
        raise e

    return 0
