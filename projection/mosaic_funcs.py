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
from skimage import exposure, color

FRAME_HEIGHT = 128
FRAME_WIDTH  = 1648

## filter ids for B, G and R 
FILTERS     = ['B','G','R']
CAMERA_IDS  = [-61501, -61502, -61503]

NLAT_SLICE  = 10


shared_lat = None
shared_lon = None
shared_img = None
shared_LON = None
shared_LAT = None
nlon       = 0
nlat       = 0

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def map_project_multi(files, pixres=1./25., num_procs=1):
    nfiles = len(files)

    lats = []
    lons = []
    incs = []
    for i, file in enumerate(files):
        dataset = nc.Dataset(file, 'r')

        lati = dataset.variables['lat'][:]
        loni = dataset.variables['lon'][:]
        inci = dataset.variables['inclination'][:]
        lats.append(lati)
        lons.append(loni)
        incs.append(inci)


    masks   = [(lats[i]!=-1000)&(lons[i]!=-1000)&(np.abs(incs[i])<np.radians(89.)) for i in range(len(files))]

    latmin = np.min([lats[i][masks[i]].min() for i in range(len(files))])
    latmax = np.max([lats[i][masks[i]].max() for i in range(len(files))])
    lonmin = np.min([lons[i][masks[i]].min() for i in range(len(files))])
    lonmax = np.max([lons[i][masks[i]].max() for i in range(len(files))])

    print("Extents - lon: %.3f %.3f  lat: %.3f %.3f"%(lonmin, lonmax, latmin, latmax))
    lats = None
    lons = None

    newlon = np.arange(lonmin, lonmax, pixres)
    newlat = np.arange(latmin, latmax, pixres)

    #LAT, LON = np.meshgrid(newlat, newlon)

    nlat = newlat.size
    nlon = newlon.size 
    
    IMG  = np.zeros((nlat, nlon, 3))
    NPIX = np.zeros((nlat, nlon, 3), dtype=np.int)

    print("Mosaic shape: %d x %d"%(nlon, nlat))

    for i, file in enumerate(files):
        fname = files[i][:-3]
        _, IMGi, mask = map_project(newlon, newlat, file, save=True, savemask=True, num_procs=num_procs)

        #NPIX[:] = NPIX[:] + mask[:]
        #IMG[:]  = IMG[:] + IMGi[:]
        IMG[:]   = np.max([IMG, IMGi], axis=0)

    #IMG[NPIX>0] = IMG[NPIX>0]/NPIX[NPIX>0]

    ## save these parameters to a NetCDF file so that we can plot it later 
    f = nc.Dataset('multi_proj_raw.nc', 'w')

    xdim     = f.createDimension('x',nlon)
    ydim     = f.createDimension('y',nlat)
    colors   = f.createDimension('colors',3)

    ##  create the NetCDF variables 
    latVar  = f.createVariable('lat', 'float64', ('y'))
    lonVar  = f.createVariable('lon', 'float64', ('x'))
    imgVar  = f.createVariable('img', 'float64', ('y','x','colors'))

    latVar[:]  = newlat[:]
    lonVar[:]  = newlon[:]
    imgVar[:]  = IMG[:]

    f.close()
    
    ## normalize the image by the 95% percentile 
    IMG = IMG/(np.percentile(IMG[IMG>0.], 95.))

    IMG = np.clip(IMG, 0., 1.)

    plt.imsave('mosaic_RGB.png', IMG, origin='lower')
    return (newlon, newlat, IMG)

def map_project(newlon, newlat, file, num_procs=1, save=False, savemask=False):
    '''
        Interpolate a single file onto a regular lon/lat grid

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
    '''
    global shared_LON, shared_LAT, nlon, nlat 

    fname   = file[:-3]
    print("Projecting %s"%fname)

    ## open the file and load the data
    dataset = nc.Dataset(file, 'r')
    lats   = dataset.variables['lat'][:]
    lons   = dataset.variables['lon'][:]
    imgs   = dataset.variables['img'][:].astype(float)
    scloci = dataset.variables['scloc'][:]
    eti    = dataset.variables['et'][:]
    incl   = dataset.variables['inclination'][:]
    emis   = dataset.variables['emission'][:]

    nframes = eti.shape[0]

    ## define the arrays to hold the new gridded data
    nlat = newlat.size
    nlon = newlon.size
    IMG  = np.zeros((nlat, nlon, 3))
    mask = np.zeros((nlat, nlon, 3))
    LAT, LON = np.meshgrid(newlat, newlon)
    
    ## create a shared memory object for LON/LAT
    LON_ctypes = np.ctypeslib.as_ctypes(LON)
    shared_LON = sct.RawArray(LON_ctypes._type_, LON_ctypes)
    LAT_ctypes = np.ctypeslib.as_ctypes(LAT)
    shared_LAT = sct.RawArray(LAT_ctypes._type_, LAT_ctypes)

    ## get the image mask where no data exists
    ## this is created to remove errors from interpolation
    output = image_mask_c(np.radians(newlat), np.radians(newlon), nlat, nlon, \
                       eti, nframes)
    maski = ctypes.cast(output, ctypes.POINTER(ctypes.c_int*(nlat*nlon))).contents
    maski = np.asarray(maski, dtype=np.int).reshape((nlat, nlon))

    ## save the mask and the raw pixel values if needed
    if(savemask):
        plt.imsave('mask_%s.png'%(fname), maski, vmin=0., vmax=1., cmap='gray', origin='lower')

    for ci in range(3):
        print("Processing %s"%(FILTERS[ci]))
        lati = lats[:,ci,:,:].flatten()
        loni = lons[:,ci,:,:].flatten()
        imgi = imgs[:,ci,:,:].flatten()
        emi  = emis[:,ci,:,:].flatten()
        inci = incl[:,ci,:,:].flatten()

        mu = np.cos(emi); mu0 = np.cos(inci)
        scorr = 2.*mu0/(mu0 + mu)

        invmask = np.where((lati==-1000.)|(loni==-1000.)|(np.abs(inci)>np.radians(89.)))[0]
        ## remove pixels that were not projected
        lat = np.delete(lati, invmask)
        lon = np.delete(loni, invmask)
        img = np.delete(imgi, invmask)
        scorr = np.delete(scorr, invmask)
        img = img/scorr

        ## do the gridding
        IMGI = project_to_uniform_grid(lon, lat, img, num_procs)
        
        ## remove interpolation errors
        IMGI[np.isnan(IMGI)]  = 0.
        IMGI[IMGI<0.] = 0.

        maski[IMGI<0.001] = 0


        IMG[:,:,ci]  = IMGI
        mask[:,:,ci] = maski
        
        if(save):
            plt.imsave("%s_%s.png"%(fname, FILTERS[2-ci]), IMGI, cmap='gray', origin='lower')

    maski = np.clip(maski, 0, 1)
    for ci in range(3):
        IMG[:,:,ci] = maski*IMG[:,:,ci]

    ## switch from BGR to RGB
    IMG = IMG[:,:,::-1]

    IMG = IMG/np.percentile(IMG, 99)
    IMG = np.clip(0., 1.)#IMG/np.percentile(IMG, 99)

    if(save):
        plt.imsave("%s_mosiac.png"%(fname), IMG, origin='lower')

    ## save these parameters to a NetCDF file so that we can plot it later
    with nc.Dataset('%s_proj.nc'%fname, 'w') as f:
        xdim     = f.createDimension('x',nlon)
        ydim     = f.createDimension('y',nlat)
        colors   = f.createDimension('colors',3)

        ## create the NetCDF variables
        latVar  = f.createVariable('lat', 'float32', ('y'), zlib=True)
        lonVar  = f.createVariable('lon', 'float32', ('x'), zlib=True)

        lonVar.units = "degrees east"
        latVar.units = "degrees north"

        imgVar  = f.createVariable('img', 'float32', ('y','x','colors'), zlib=True)
        
        img_corrVar = f.createVariable('img_corr', 'uint8', ('y','x','colors'), zlib=True)

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]
        img_corrVar[:] = np.asarray(IMG*255/IMG.max(), dtype=np.uint8)

    return "%s_proj.nc"%fname, IMG, mask


def project_to_uniform_grid(lon, lat, img, num_procs=1):
    '''
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

    '''
    global shared_img, shared_lon, shared_lat, shared_LON, shared_LAT, nlon, nlat
    nsquare_lon     = int(np.ceil(nlon/num_procs))
    nsquare_lat     = int(np.ceil(nlat/NLAT_SLICE))

    ## conver the data arrays into shared memory
    lon_ctypes = np.ctypeslib.as_ctypes(lon)
    shared_lon = sct.RawArray(lon_ctypes._type_, lon_ctypes)
    lat_ctypes = np.ctypeslib.as_ctypes(lat)
    shared_lat = sct.RawArray(lat_ctypes._type_, lat_ctypes)
    img_ctypes = np.ctypeslib.as_ctypes(img)
    shared_img = sct.RawArray(img_ctypes._type_, img_ctypes)

    inpargs = []
    indices = []
    
    ## convert back to a numpy array to process
    LON = np.asarray(shared_LON, dtype=np.float32).reshape(nlon,nlat)
    LAT = np.asarray(shared_LAT, dtype=np.float32).reshape(nlon,nlat)

    ## build the inputs to the multiprocessing pipeline
    ## this will decompose the longitude grid into num_procs
    ## and the latitude grid into NLAT_SLICEs
    for i in range(num_procs):
        startxind = i*nsquare_lon
        endxind   = min([nlon, (i+1)*nsquare_lon])
        for j in range(NLAT_SLICE):
            startyind = j*nsquare_lat
            endyind   = min([nlat, (j+1)*nsquare_lat])
            LONi = LON[startxind:endxind,startyind:endyind]
            LATi = LAT[startxind:endxind,startyind:endyind]
            
            lonmin   = LONi.min()
            lonmax   = LONi.max()
            latmin   = LATi.min()
            latmax   = LATi.max()

            maski    = np.where((lon>lonmin-2.)&(lon<lonmax+2.)&\
                                (lat>latmin-2)&(lat<latmax+2))[0]

            loni     = lon[maski]
            lati     = lat[maski]
            imgi     = img[maski]

            ## make sure there is enough data to grid 
            if(len(loni) > 3):
                inpargs.append([startxind,endxind,startyind,endyind, maski])

    ## create the final image array
    IMG = np.zeros((nlat,nlon),dtype=np.float64)

    ## start the pool
    pool = multiprocessing.Pool(num_procs, initializer=initializer)
    try:
        i = 0

        ## start the multicore grid processing
        for ri in pool.imap_unordered(project_part_image, inpargs):
            progress = i/len(inpargs)
            print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
            time.sleep(0.05)

            ## construct the image array from the output
            startxind, endxind, startyind, endyind, IMGi = ri
            IMG[startyind:endyind,startxind:endxind] = IMGi
            i += 1 
        pool.close()

    except Exception as e:
        pool.terminate()
        pool.join()
        raise e
        sys.exit()
    
    print()
    pool.join()

    return IMG

def project_part_image(inp, method='linear'):
    '''
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
    '''
    global shared_lon, shared_lat, shared_img, shared_LON, shared_LAT, nlon, nlat
    
    startxind,endxind,startyind,endyind, maski = inp
    
    lon = np.asarray(shared_lon, dtype=np.float32)[maski]
    lat = np.asarray(shared_lat, dtype=np.float32)[maski]
    img = np.asarray(shared_img, dtype=np.float64)[maski]
    LON = np.asarray(shared_LON, dtype=np.float32).\
        reshape(nlon,nlat)[startxind:endxind,startyind:endyind]
    LAT = np.asarray(shared_LAT, dtype=np.float32).\
        reshape(nlon,nlat)[startxind:endxind,startyind:endyind]

    try:
        imgi =  griddata((lon, lat), \
                         img, (LON, LAT), method=method).T
        return (startxind, endxind, startyind, endyind, imgi)
    except Exception as e:
        raise e

def color_correction(datafile, gamma=1.0, hist_eq=True, fname=None, save=False):
    ''' 
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

        Outputs
        -------
        IMG2 : numpy.ndarray
            Output color and gamma corrected file with 99-percentile scaling

        Raises
        ------
        AssertionError
            if `save`=True but no fname defined
    '''
    with nc.Dataset(datafile, "r+") as data:
        IMG  = data.variables['img'][:]
        lon  = data.variables['lon'][:]
        lat  = data.variables['lat'][:]

        IMG2 = IMG.copy()
        ## cleanup and do color correction
        IMG2[:,:,0] *= 0.902
        IMG2[:,:,2] *= 1.8879

        if hist_eq:
            ## normalize the image by the 95% percentile
            IMG2 = IMG2/(np.percentile(IMG2[IMG2>0.], 99.))
            IMG2 = np.clip(IMG2, 0, 1)
            hsv  = color.rgb2hsv(IMG2)
            val = hsv[:,:,2].flatten()
            val = val[val>0.]
            p0, p1 = np.percentile(val, (2,99))
            hsv[:,:,2] = exposure.rescale_intensity(hsv[:,:,2], in_range=(p0, p1))
            #hsv[:,:,2] = exposure.equalize_adapthist(hsv[:,:,2], clip_limit=0.05)
            IMG2 = color.hsv2rgb(hsv)

        IMG2 = IMG2**gamma
        ## normalize the image by the 95% percentile
        IMG2 = IMG2/(np.percentile(IMG2[IMG2>0.], 99.))
        IMG2 = np.clip(IMG2, 0, 1)

        ## save the new image out to the netCDF file
        if 'img_corr' not in data.variables.keys():
            img_corrVar = data.createVariable('img_corr', 'uint8', ('y','x','colors'), zlib=True)
            img_corrVar[:] = np.asarray(IMG2*255, dtype=np.uint8)
        else:
            data.variables['img_corr'][:] = np.asarray(IMG2*255, dtype=np.uint8)

    if save:
        assert not isinstance(fname, type(None)), \
            "please provide a filename to save the data"
        plt.imsave('%s_mosaic_RGB.png'%fname, IMG2, origin='lower')
    

    fig, ax = plt.subplots(1,1,figsize=(10,10),dpi=150)
    ax.imshow(IMG2, extent=(lon.min(), lon.max(), lat.min(), lat.max()),\
               origin='lower')
    ax.set_xlabel(r"Longitude [deg]")
    ax.set_ylabel(r"Latitude [deg]")
    plt.show()

    return IMG2


def plot_ortho(datafile, sat_height_scale=1., facecolor='black'):
    with nc.Dataset(datafile, "r") as data:
        if 'img_corr' in data.variables.keys():
            IMG  = data.variables['img_corr'][:]
        else:
            IMG  = data.variables['img'][:]
        lon  = data.variables['lon'][:]
        lat  = data.variables['lat'][:]

    LON, LAT = np.meshgrid(lon, lat)

    proj = ccrs.NearsidePerspective(central_longitude=LON[IMG.min(axis=2)>0.].mean(), \
                                    central_latitude=LAT[IMG.min(axis=2)>0.].mean(), \
                                    satellite_height=35785.e3*sat_height_scale)

    fig = plt.figure(figsize=(10,10), dpi=150, facecolor=facecolor)
    ax  = fig.add_subplot(projection=proj)

    ax.imshow(IMG[::-1,:], origin='upper', extent=(lon.min(), lon.max(), lat.min(), lat.max()), \
              transform=ccrs.PlateCarree(), interpolation='hermite')

    plt.show()

