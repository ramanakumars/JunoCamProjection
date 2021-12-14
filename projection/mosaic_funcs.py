from .camera_funcs import *

NLAT_SLICE  = 20
NLON_SLICE  = 20

BOX_X = 20
BOX_Y = 20

shared_lat = None
shared_lon = None
shared_img = None
shared_LON = None
shared_LAT = None
shared_IMG = None
nlon       = 0
nlat       = 0
nfiles     = 0

shared_IMGs  = None
shared_Ls    = None
shared_INCDs = None
shared_incem = None
ave_Ls       = 0.

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def map_project_multi(files, pixres=1./25., num_procs=1, extents=None, \
                      scorr_method='fft', load=False, **kwargs):
    '''
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
            bounding box in the format [lonmin, lonmax, latmin, latmax] (in degrees)
            set to None to let it automatically determine the extents from the extents
            of all the images
        scorr_method : string
            lighting correction method to use:
                - 'none': no correction
                - 'simple': lambertian correction (output = input/cos(incidence))
                - 'poly': fit a 5th order polynomial for the image brightness in (mu, mu0) space
                - 'fft': use an FFT to correct for brightness variations in the image
        load : bool
            Set to `True` to load data from files (generated from previous run). 
            Set to `False` to regenerate the image.
        kwargs: arguments to pass to `map_project`
    '''

    nfiles = len(files)

    lats = []
    lons = []
    incs = []

    # Load all the files and get the lat/lon
    # info as well as the image and lighting geometry
    for i, file in enumerate(files):
        with nc.Dataset(NC_FOLDER+file, 'r') as dataset:
            lati = dataset.variables['lat'][:]
            loni = dataset.variables['lon'][:]
            inci = dataset.variables['incidence'][:]
            
            lon_rot = kwargs.get('lon_rot', 0.)
            loni[loni!=-1000] += lon_rot
            loni[(loni < -180.)&(loni!=-1000)] += 360.
            loni[loni > 180.]  -= 360.

            lats.append(lati)
            lons.append(loni)
            incs.append(inci)

    # mask out bad pixels
    masks   = [(lats[i]!=-1000)&(lons[i]!=-1000)&(np.abs(incs[i])<np.radians(89.)) for i in range(len(files))]

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

    print("Extents - lon: %.3f %.3f  lat: %.3f %.3f"%(lonmin, lonmax, latmin, latmax), flush=True)
    lats = None
    lons = None
    incs = None

    # create the uniform grid for mosaicing
    newlon = np.arange(lonmin, lonmax, pixres)
    newlat = np.arange(latmin, latmax, pixres)

    #LAT, LON = np.meshgrid(newlat, newlon)

    nlat = newlat.size
    nlon = newlon.size 
    
    # create the arrays to hold the intermediary data
    IMG   = np.zeros((nlat, nlon, 3), dtype=np.float32)
    IMGs  = np.zeros((len(files), nlat, nlon, 3), dtype=np.float32)
    Ls    = np.zeros((len(files), nlat, nlon), dtype=np.float32)
    INCDs = np.zeros((len(files), nlat, nlon), dtype=np.float32)
    EMISs = np.zeros((len(files), nlat, nlon), dtype=np.float32)

    print("Mosaic shape: %d x %d"%(nlon, nlat), flush=True)

    for i, file in enumerate(files):
        # generate the projection of each image in the mosaic
        fname = files[i][:-3]
        _, IMGi, _, INCDs[i,:], EMISs[i,:] = map_project(file, pixres=pixres, long=newlon, latg=newlat,
                        save=True, savemask=True, num_procs=num_procs, interp_loaded_grid=True,
                        scorr_method=scorr_method, load=load, ret_inc=True, **kwargs)

        IMGi[np.isnan(IMGi)] = 0.
        # save the data and also get brightness information
        IMGs[i,:] = IMGi
        Ls[i,:]   = color.rgb2hsv(IMGi)[:,:,2]
        sys.stdout.flush()

    del IMGi
    gc.collect()

    combine_method = kwargs.get('combine_method', 'max')

    # experimental method (not recommended)
    if combine_method=='min_grad':
        Lx, Ly = np.gradient(Ls, axis=(1,2))
        Lxx = np.gradient(Lx, axis=1)
        Lyy = np.gradient(Ly, axis=2)
        delsqL = np.abs(Lx + Ly)

        for n in range(delsqL.shape[0]):
            delsqL[n,:,:] = get_fft(delsqL[n,:,:], 500)

    # get the overlap area, and correct for 
    # brightness variations between overlap
    # regions in different images
    npix    = np.sum(Ls>0.005*Ls.max(), axis=0)
    overlap_mask = npix>1

    # get the average value of the overlap
    # region in each image
    ave_val = np.zeros(len(files))
    for i in range(len(files)):
        imgi = Ls[i,:][overlap_mask]
        ave_val[i] = (imgi[imgi>0.]).mean()

    # get the average of the all overlaps
    ave_all = np.mean(ave_val)
    print(ave_val, ave_all)

    # correct each image so that the overlap 
    # regions have the same brightness
    for i in range(len(files)):
        IMGs[i,:] *= ave_all/ave_val[i]
        Ls[i,:]    = color.rgb2hsv(IMGs[i,:])[:,:,2]
    
    # clear the memory 
    del overlap_mask
    del ave_val
    del npix
    gc.collect()

    sys.stdout.flush()

    if combine_method=='max':
        combine = np.max
    elif combine_method=='min':
        combine = np.min

    if 'none' in scorr_method:
        IMG = np.max(IMGs, axis=0)
        incem = None
    else:
        # create a dummy lighting geometry variable
        incem = INCDs*EMISs

        # remove regions of low lighting (even after 
        # lighting correction)
        incem[IMGs.sum(axis=-1)<0.02] = np.nan
        print("Mosaicing image", flush=True)

        # loop through every pixel and find the best value to assign to the mosaic
        if combine_method in ['min', 'max']:
            for jj in range(IMG.shape[0]):
                if jj%100==0:
                    print("\r[%-20s] %d/%d"%(int(jj/IMG.shape[0]*20.)*'=', jj+1, IMG.shape[0]), 
                          end='', flush=True)
                for ii in range(IMG.shape[1]):
                    incemij = incem[:,jj,ii]
                    
                    # the best pixel is one that actually saw that feature, is not too dim, 
                    # and does not have a low incidence angle
                    mask = ~np.isnan(incemij)&(Ls[:,jj,ii] > 0.5*Ls[:,jj,ii].max())&\
                        (INCDs[:,jj,ii]<np.radians(80))
                    nimgs = np.sum(mask)
                    if nimgs > 1:
                        # set the value of the pixel in the mosaic
                        if combine_method=='min_grad':
                            IMG[jj,ii,:] = IMGs[np.argmin(delsqL[mask,jj,ii]),jj,ii,:]
                        else:
                            IMG[jj,ii,:] = combine(IMGs[mask,jj,ii,:], axis=0)
                    elif nimgs > 0:
                        IMG[jj,ii,:] = IMGs[mask,jj,ii,:]
        elif combine_method=='box_average_old':
            nx = int(np.ceil(IMG.shape[1]/BOX_X))
            ny = int(np.ceil(IMG.shape[0]/BOX_Y))

            for jj in range(ny):
                if jj%100==0:
                    print("\r[%-20s] %d/%d"%(int(jj/ny*20.)*'=', jj+1, ny), end='', flush=True)
                starty = jj*BOX_Y
                endy   = min([IMG.shape[0], (jj+1)*BOX_Y])
                for ii in range(nx):
                    startx  = ii*BOX_X
                    endx    = min([IMG.shape[1], (ii+1)*BOX_X])

                    alpha = np.zeros((IMGs.shape[0], BOX_Y, BOX_X))
                    # get the images in this box
                    incem_ij = incem[:,starty:endy,startx:endx]
                    incds_ij = INCDs[:,starty:endy,startx:endx]
                    Ls_ij    = Ls[:,starty:endy,startx:endx]
                    imgs_ij  = IMGs[:,starty:endy,startx:endx]

                    # check if there any any images in this box
                    mask = ~np.isnan(incem_ij)&(Ls_ij > 0.5*Ls_ij.max())&\
                        (incds_ij<np.radians(80))
                    if np.sum(mask) > 1:
                        # if there are, then assign weights (alpha) to each pixel in 
                        # each image. final image is a linear combination of images
                        # with alphas
                        for kk in range(IMGs.shape[0]):
                            mask = ~np.isnan(incem_ij[kk,:])&(Ls_ij[kk,:] > 0.5*Ls[kk,:].max())&\
                                (incds_ij[kk,:]<np.radians(80))

                            # weight each image by its relative brightness wrt to the 
                            # global mean
                            if len(Ls_ij[kk,:,:][mask]) > 0:
                                alpha[kk,:,:][mask]  = Ls_ij[kk,:,:][mask].mean()/ave_all
                                alpha[kk,:][alpha[kk,:]!=0] = 1./alpha[kk,:][alpha[kk,:]!=0.]

                        for c in range(3):
                            IMGs_sub = np.multiply(imgs_ij[:,:,:,c], alpha)
                            alpha_sum = np.sum(alpha, axis=0)
                            alpha_sum[alpha_sum==0] = np.nan
                            IMG[starty:endy,startx:endx,c] = np.divide(np.sum(IMGs_sub, axis=0), 
                                                                        alpha_sum)
                            IMG[np.isnan(IMG)] = 0.
                    else:
                        continue
        else:
            IMG = box_average(IMGs, INCDs, incem, Ls, ave_all, num_procs=num_procs)
    # clean up 
    del INCDs
    del EMISs

    # save these parameters to a NetCDF file so that we can plot it later 
    with nc.Dataset(NC_FOLDER+'multi_proj_raw.nc', 'w') as f:
        xdim     = f.createDimension('x',nlon)
        ydim     = f.createDimension('y',nlat)
        colors   = f.createDimension('colors',3)
        files    = f.createDimension('file',len(files))

        ##  create the NetCDF variables 
        latVar   = f.createVariable('lat', 'float64', ('y'), zlib=True)
        lonVar   = f.createVariable('lon', 'float64', ('x'), zlib=True)
        imgVar   = f.createVariable('img', 'float64', ('y','x','colors'), zlib=True)
        imgsVar  = f.createVariable('imgs', 'float32', ('file', 'y','x','colors'), zlib=True)
        if incem is not None:
            incemVar = f.createVariable('incem', 'float64', ('file', 'y','x'), zlib=True)

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]
        imgsVar[:] = IMGs[:]
        if incem is not None:
            incemVar[:] = incem[:]
    
    ## normalize the image by the 99% percentile 
    IMG = IMG/(np.percentile(IMG[IMG>0.], 99.))
    IMG = np.clip(IMG, 0., 1.)

    plt.imsave(MOS_FOLDER+'mosaic_RGB.png', IMG, origin='lower')
    return (newlon, newlat, IMG)


def map_project(file, long=None, latg=None, pixres=None, num_procs=1, \
                save=False, savemask=False, scorr_method='simple', load=False, ret_inc=False,
                **kwargs):
    '''
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
    '''
    from scipy.interpolate import RegularGridInterpolator
    global shared_LON, shared_LAT, nlon, nlat 

    fname   = file[:-3]
    print("Projecting %s"%fname)

    if not os.path.exists(MOS_FOLDER):
        os.mkdir(MOS_FOLDER)
    if not os.path.exists(MASK_FOLDER):
        os.mkdir(MASK_FOLDER)
    if not os.path.exists(NPY_FOLDER):
        os.mkdir(NPY_FOLDER)

    ## open the file and load the data
    dataset = nc.Dataset(NC_FOLDER+file, 'r')
    lats   = dataset.variables['lat'][:]
    lons   = dataset.variables['lon'][:]
    use_flux = kwargs.get('use_flux', False)
    if use_flux:
        imgs   = dataset.variables['flux'][:].astype(float)
    else:
        print("Using raw IMG")
        imgs   = dataset.variables['img'][:].astype(float)
    scloci = dataset.variables['scloc'][:]
    eti    = dataset.variables['et'][:]
    incid   = dataset.variables['incidence'][:]
    emis   = dataset.variables['emission'][:]

    # rotate the lon grid if needed
    lon_rot = kwargs.get('lon_rot', 0.)
    lons[lons!=-1000] += lon_rot
    lons[(lons < -180.)&(lons!=-1000)] += 360.
    lons[lons > 180.]  -= 360.

    nframes = eti.shape[0]

    if (pixres is not None):
        masks   = (lats!=-1000)&(lons!=-1000)&(np.abs(incid)<np.radians(89.))
        latmin = lats[masks].min()
        latmax = lats[masks].max()
        lonmin = lons[masks].min()
        lonmax = lons[masks].max()

        newlon = np.arange(lonmin, lonmax, pixres)
        newlat = np.arange(latmin, latmax, pixres)
        print("Limits: lon: %.3f %.3f  lat: %.3f %.3f  size: %d x %d"%(\
                newlon.min(), newlon.max(), newlat.min(), newlat.max(), newlon.size, newlat.size))
    else:
        raise RuntimeError("Please provide a resolution")

    ## define the arrays to hold the new gridded data
    nlat = newlat.size
    nlon = newlon.size
    IMG  = np.zeros((nlat, nlon, 3))
    LAT, LON = np.meshgrid(newlat, newlon)

    ## get the image mask where no data exists
    ## this is created to remove errors from interpolation
    if load&os.path.exists(NPY_FOLDER+"%s_mask.npy"%fname):
        print("Loading mask file")
        maski = np.load(NPY_FOLDER+"%s_mask.npy"%fname)
    else:
        roll_lon = newlon - lon_rot
        roll_lon[roll_lon < -180.] += 360.
        roll_lon[roll_lon >  180.] -= 360.
        output = image_mask_c(np.radians(newlat), np.radians(roll_lon), nlat, nlon, \
                           eti, nframes)
        maski = ctypes.cast(output, ctypes.POINTER(ctypes.c_int*(nlat*nlon))).contents
        maski = np.asarray(maski, dtype=np.int).reshape((nlat, nlon))
        np.save(NPY_FOLDER + "%s_mask.npy"%fname, maski)
    
    ## save the mask and the raw pixel values if needed
    if(savemask):
        plt.imsave(MASK_FOLDER+'mask_%s.png'%(fname), maski, vmin=0., vmax=1., cmap='gray', origin='lower')

    # loop through each color and create that color band in the projection
    for ci in range(3):
        print("Processing %s"%(FILTERS[ci]))
        filteriname = NPY_FOLDER + "%s_%s.npy"%(fname, FILTERS[ci])
        # load the image if needed
        if load&os.path.exists(filteriname):
            print("Loading from %s"%filteriname)
            IMGI = np.load(filteriname)
        else:
            ## create a shared memory object for LON/LAT
            LON_ctypes = np.ctypeslib.as_ctypes(LON)
            shared_LON = sct.RawArray(LON_ctypes._type_, LON_ctypes)
            LAT_ctypes = np.ctypeslib.as_ctypes(LAT)
            shared_LAT = sct.RawArray(LAT_ctypes._type_, LAT_ctypes)

            lati = lats[:,ci,:,:].flatten()
            loni = lons[:,ci,:,:].flatten()
            imgi = imgs[:,ci,:,:].flatten()
            emi  = emis[:,ci,:,:].flatten()
            inci = incid[:,ci,:,:].flatten()

            invmask = np.where((lati==-1000.)|(loni==-1000.)|(np.abs(inci)>np.radians(85.)))[0]
            ## remove pixels that were not projected
            lat = np.delete(lati, invmask)
            lon = np.delete(loni, invmask)
            img = np.delete(imgi, invmask)
            
            ## do the gridding
            IMGI = project_to_uniform_grid(lon, lat, img, num_procs)
            
            ## remove interpolation errors
            IMGI[np.isnan(IMGI)]  = 0.
            IMGI[IMGI<0.] = 0.

            ## Save the data out to a numpy file
            np.save(filteriname, IMGI)
        maski[IMGI<0.001] = 0
        IMG[:,:,ci]  = IMGI

    INCD, EMIS = get_emis_incid_map(incid, emis, lats, lons, \
                                   newlat, newlon, maski, fname, \
                                   num_procs=num_procs,\
                                   load=load)
    maski = np.clip(maski, 0, 1)
    for ci in range(3):
        IMG[:,:,ci] = maski*IMG[:,:,ci]

    ## switch from BGR to RGB
    IMG = IMG[:,:,::-1]
    
    ## Do a simple Lambertian correction if needed
    if 'simple' in scorr_method:
        mu0 = np.clip(np.cos(INCD), 0, 1)
        mu0[mu0<0.01] = np.nan
        scorr = mu0#2.*mu0/(mu0 + mu)
        for ci in range(3):
            IMG[:,:,ci] = IMG[:,:,ci]/scorr


    if 'poly' in scorr_method:
        IMG = scorr_poly(INCD, EMIS, newlat, newlon, IMG)

    # with both the poly and simple corrections,
    # bad incidence values are treated as NaNs. 
    # remove those pixels here
    IMG[np.isnan(IMG)] = 0.

    if 'fft' in scorr_method:
        # do an FFT correction if needed
        IMG = scorr_fft(IMG, fname, \
                        radius=kwargs.get('fft_radius', 5.), \
                        colorspace=kwargs.get('colorspace', 'hsv'), \
                        trim_rad=kwargs.get('trim_rad', 0.6), 
                        trim_threshold=kwargs.get('trim_thresh', 0.95))
        

    ## expand the image out to the required grid
    interp_loaded_grid=kwargs.get('interp_loaded_grid', False)
    if interp_loaded_grid:
        assert (latg is not None)&(long is not None), "Please provide the grid lat/lon"
        IMGnew = np.zeros((latg.size, long.size, 3))
        LATG, LONG = np.meshgrid(latg, long)
        newpoints = np.stack((LATG.flatten(), LONG.flatten()), axis=1).reshape(-1, 2)
        print("Interpolating from %d x %d => %d x %d"%(nlat, nlon, latg.size, long.size))
        for ci in range(3):
            interp_function = RegularGridInterpolator((newlat, newlon), IMG[:,:,ci], 
                                                      bounds_error=False, fill_value=0.)
            imgi = interp_function(newpoints).reshape((long.size, latg.size)).T
            IMGnew[:,:,ci] = imgi

        # remove interpolation errors where one color has dark data but the
        # other don't. these needs to be large enough to ignore dark areas,
        # but small enough that we don't miss color features (such as GRS)
        IMGnew = IMGnew.reshape((long.size*latg.size, 3))
        IMGnew[IMGnew.min(axis=-1)<0.1,:] = 0
        IMG = IMGnew.reshape((latg.size, long.size, 3))
        imgi = IMG.copy()
        
        trim_size = 2
        # trim edges from this image to avoid interpolation errors
        for jj in range(trim_size,latg.size-trim_size):
            if os.environ.get('NO_VERBOSE') is None:
                print("\rTrimming: [%-20s] %d/%d"%(int(jj/latg.size*20)*'=', jj, latg.size), end='')
            # ignore if this column has no data (speeds up trim computation)
            if IMG[jj,:,:].max()==0:
                continue
            for ii in range(trim_size, long.size-trim_size):
                if IMG[jj,ii,:].mean() == 0:
                    continue
                # get the window around this pixel
                img_sub = IMG[(jj-trim_size):(jj+trim_size),(ii-trim_size):(ii+trim_size),:]

                # if there is an edge in this image (given by a 0 value)
                # set the pixel value to 0
                if img_sub.mean(axis=-1).min() < 0.1:
                    imgi[jj,ii,:] = 0.
        print()

        IMG = imgi.copy()

        # interpolate the other variables
        interp_function = RegularGridInterpolator((newlat, newlon), INCD, 
                                                  bounds_error=False, fill_value=np.nan)
        INCD = interp_function(newpoints).reshape((long.size, latg.size)).T
        interp_function = RegularGridInterpolator((newlat, newlon), EMIS,
                                                  bounds_error=False, fill_value=np.nan)
        EMIS = interp_function(newpoints).reshape((long.size, latg.size)).T
        interp_function = RegularGridInterpolator((newlat, newlon), maski,
                                                  bounds_error=False, fill_value=0.)
        maski = interp_function(newpoints).reshape((long.size, latg.size)).T

        # update the lat/lon info for output
        newlat = latg
        newlon = long
        nlon   = long.size
        nlat   = latg.size

    IMGc = IMG/IMG.max()
    IMGc = np.clip(IMGc, 0., 1.)#IMG/np.percentile(IMG, 99)

    if(save):
        plt.imsave(MOS_FOLDER+"%s_mosaic.png"%(fname), IMGc, origin='lower')

    ## save these parameters to a NetCDF file so that we can plot it later
    with nc.Dataset(NC_FOLDER+'%s_proj.nc'%fname, 'w') as f:
        xdim     = f.createDimension('x',nlon)
        ydim     = f.createDimension('y',nlat)
        colors   = f.createDimension('colors',3)

        ## create the NetCDF variables
        latVar  = f.createVariable('lat', 'float32', ('y'), zlib=True)
        lonVar  = f.createVariable('lon', 'float32', ('x'), zlib=True)

        lonVar.units = "degrees east"
        latVar.units = "degrees north"

        imgVar  = f.createVariable('img', 'float64', ('y','x','colors'), zlib=True)
        
        img_corrVar = f.createVariable('img_corr', 'uint8', ('y','x','colors'), zlib=True)

        incdVar = f.createVariable('incd', 'float32', ('y', 'x'), zlib=True)
        emisVar = f.createVariable('emis', 'float32', ('y', 'x'), zlib=True)

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]
        incdVar[:]  = INCD[:]
        emisVar[:]  = EMIS[:]
        img_corrVar[:] = np.asarray(IMGc*255/IMG.max(), dtype=np.uint8)

    if ret_inc:
        return "%s_proj.nc"%fname, IMG, maski, INCD, EMIS
    else:
        return "%s_proj.nc"%fname, IMG, maski


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
    global shared_img, shared_lon, shared_lat, shared_LON, shared_LAT, shared_IMG,\
         nlon, nlat
    nsquare_lon     = int(np.ceil(nlon/NLON_SLICE))
    nsquare_lat     = int(np.ceil(nlat/NLAT_SLICE))

    ## conver the data arrays into shared memory
    '''
    lon_ctypes = np.ctypeslib.as_ctypes(lon)
    shared_lon = sct.RawArray(lon_ctypes._type_, lon_ctypes)
    lat_ctypes = np.ctypeslib.as_ctypes(lat)
    shared_lat = sct.RawArray(lat_ctypes._type_, lat_ctypes)
    img_ctypes = np.ctypeslib.as_ctypes(img)
    shared_img = sct.RawArray(img_ctypes._type_, img_ctypes)
    '''
    inpargs = []
    indices = []
    
    ## convert back to a numpy array to process
    LON = np.asarray(shared_LON, dtype=np.float32).reshape(nlon,nlat)
    LAT = np.asarray(shared_LAT, dtype=np.float32).reshape(nlon,nlat)

    ## get the pixel resolution in deg/pixel 
    ## used as a search radius to get the points within the
    ## domain bounds
    pixres= LON[1,0] - LON[0,0]

    ## build the inputs to the multiprocessing pipeline
    ## this will decompose the longitude grid into NLON_SLICE
    ## and the latitude grid into NLAT_SLICEs
    for j in range(NLAT_SLICE):
        startyind = j*nsquare_lat
        endyind   = min([nlat, (j+1)*nsquare_lat])
        print("\r %d/%d"%(j, NLAT_SLICE), end='')
        for i in range(NLON_SLICE):
            startxind = i*nsquare_lon
            endxind   = min([nlon, (i+1)*nsquare_lon])
            LONi = LON[startxind:endxind,startyind:endyind]
            LATi = LAT[startxind:endxind,startyind:endyind]
            
            lonmin   = LONi.min()
            lonmax   = LONi.max()
            latmin   = LATi.min()
            latmax   = LATi.max()

            maski    = np.where((lon>lonmin-50*pixres)&(lon<lonmax+50*pixres)&\
                                (lat>latmin-50*pixres)&(lat<latmax+50*pixres))[0]
            ## make sure there is enough data to grid 
            if(len(maski) > 3):
                #inpargs.append([startxind,endxind,startyind,endyind, maski])
                inpargs.append([startxind,endxind,startyind,endyind,lon[maski],lat[maski],img[maski]])
    print()
    ## create the final image array
    ## this will be stored as a shared array so each process 
    ## can write to it
    cdtype = np.ctypeslib.as_ctypes_type(np.dtype(float))
    shared_IMG = multiprocessing.RawArray(cdtype, nlat*nlon)

    ## start the pool
    pool = multiprocessing.Pool(processes=num_procs, initializer=initializer)
    try:
        i = 0

        ## start the multicore grid processing
        r = pool.map_async(project_part_image, inpargs)
        pool.close()

        tasks = pool._cache[r._job]
        ninpt = len(inpargs)
        while tasks._number_left > 0:
            progress = (ninpt - tasks._number_left*tasks._chunksize)/ninpt
            if os.environ.get('NO_VERBOSE') is None:
                print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
            time.sleep(0.05)
    except Exception as e:
        pool.terminate()
        pool.join()
        raise e
        sys.exit()
    
    print()
    pool.join()

    return np.frombuffer(shared_IMG, dtype=float).reshape((nlat,nlon)) #np.ctypeslib.as_array(IMG_ctypes)

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
    global shared_lon, shared_lat, shared_img, shared_LON, shared_LAT, shared_IMG, \
        nlon, nlat
    
    #startxind,endxind,startyind,endyind, maski = inp
    startxind,endxind,startyind,endyind,lon,lat,img = inp
    
    #lon = np.asarray(shared_lon, dtype=np.float32)[maski]
    #lat = np.asarray(shared_lat, dtype=np.float32)[maski]
    #img = np.asarray(shared_img, dtype=np.float64)[maski]
    LON = np.asarray(shared_LON, dtype=np.float32).\
        reshape(nlon,nlat)[startxind:endxind,startyind:endyind]
    LAT = np.asarray(shared_LAT, dtype=np.float32).\
        reshape(nlon,nlat)[startxind:endxind,startyind:endyind]

    IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat, nlon))#np.ctypeslib.as_array(shared_IMG)

    try:
        imgi =  griddata((lon, lat), \
                         img, (LON, LAT), method=method).T
        IMG[startyind:endyind,startxind:endxind] = imgi
        #return (startxind, endxind, startyind, endyind, imgi)
    except Exception as e:
        raise e

def color_correction(datafile, gamma=1.0, hist_eq=True, fname=None, save=False, \
                     trim_saturated=False, sat_threshold=95, clip_limit=0.008, **kwargs):
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
    '''
    with nc.Dataset(datafile, "r+") as data:
        IMG  = data.variables['img'][:]
        lon  = data.variables['lon'][:]
        lat  = data.variables['lat'][:]

        IMG2 = IMG.copy()

        if trim_saturated:
            scaling = np.percentile(IMG2[IMG2>0.], sat_threshold)
            IMGc = IMG2/scaling
            IMGc[IMGc.sum(axis=2)>2.98] = 0.
            IMG2 = IMGc*scaling

        if hist_eq:
            ## normalize the image by the 95% percentile
            IMG2 = IMG2/(np.percentile(IMG2[IMG2>0.], 99.9))
            IMG2 = np.clip(IMG2, 0, 1)
            for ci in range(3):
                IMG2[:,:,ci] = exposure.equalize_adapthist(IMG2[:,:,ci], clip_limit=clip_limit)

        IMG2 = IMG2**gamma
        ## normalize the image by the 99.9% percentile
        IMG2 = IMG2/(np.percentile(IMG2[IMG2>0.], 99.9))
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
        plt.imsave(MOS_FOLDER+'%s_mosaic_RGB.png'%fname, IMG2, origin='lower')
    

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

def get_emis_incid_map(incid, emis, lat, lon, newlat, newlon, maski, \
                      fname, num_procs=1, load=True):
    ## flatten and mask out the bad data points
    incid = incid.flatten(); emis = emis.flatten();
    lat  = lat.flatten();  lon  = lon.flatten()
    mask = (lat!=-1000)&(lon!=-1000)&(np.abs(incid)<np.radians(89.))

    incidsf  = incid[mask].flatten()
    emisf   = emis[mask].flatten()
    lonf    = lon[mask].flatten()
    latf    = lat[mask].flatten()

    ## create the new grid to project onto and
    ## project the incidence and emission values onto the new grid
    LAT, LON = np.meshgrid(newlat, newlon)

    if load&os.path.exists(NPY_FOLDER+"%s_emis.npy"%fname):
        print("Loading emission data")
        EMIS = np.load(NPY_FOLDER+"%s_emis.npy"%fname)
    else:
        print("Processing emission angles")
        EMIS = project_to_uniform_grid(lonf, latf, emisf, num_procs=num_procs)
        EMIS[~maski] = np.nan
        np.save(NPY_FOLDER+"%s_emis.npy"%fname, EMIS)
    
    if load&os.path.exists(NPY_FOLDER+"%s_incid.npy"%fname):
        print("Loading incidence data")
        INCD = np.load(NPY_FOLDER+"%s_incid.npy"%fname)
    else:
        print("Processing incidence angles")
        INCD = project_to_uniform_grid(lonf, latf, incidsf, num_procs=num_procs)
        INCD[~maski] = np.nan
        np.save(NPY_FOLDER+"%s_incid.npy"%fname, INCD)

    return (INCD, EMIS)

def scorr_poly(INCD, EMIS, newlat, newlon, IMG):
    '''
        Fit a 2nd order polynomial in incidence/emission space
        to correct for lighting geometry
    '''
    print("Doing polynomial lighting correction")


    mask = (~np.isnan(EMIS))&(~np.isnan(INCD))&(IMG.min(axis=2)>0.)

    ## get the brightness value of the pixels
    HSV = color.rgb2hsv(IMG/IMG.max())
    IMG_val = HSV[:,:,2]
    print(IMG_val.min(), IMG_val.max())
    #IMG_val = IMG[:,:,1]

    
    INCDf = np.cos(INCD[mask].flatten())
    EMISf = np.cos(EMIS[mask].flatten())
    VALf  = IMG_val[mask].flatten()

    ## remove boundary data
    mask = (INCDf>0.05)
    INCDf = INCDf[mask]
    EMISf = EMISf[mask]
    VALf  = VALf[mask]

    ind = np.asarray(range(len(INCDf)))
    np.random.shuffle(ind)
    ind = ind[:1000]

    INCDf = INCDf[ind]; EMISf = EMISf[ind]; VALf = VALf[ind]

    # create the two axis (x is mu, y is mu0)
    x  = INCDf; y = EMISf
    xx = np.cos(INCD); yy = np.cos(EMIS)

    m = 5 ## polynomial order
    A = []
    for i in range(m):
        for j in range(i):
            A.append( (x**i)*(y**j) )
    A = np.asarray(A).T

    # fit the polynomial
    coeff, r, rank, s = np.linalg.lstsq(A, VALf, rcond=None)

    # create the correction from the new polynomial
    SCORR = np.zeros_like(INCD)
    k = 0
    for i in range(m):
        for j in range(i):
            SCORR += coeff[k]*(xx**i)*(yy**j)
            k += 1

    # ignore negative values
    SCORR[SCORR<0.] = np.nan

    ## correct the image and normalize
    HSV[:,:,2] = HSV[:,:,2]/SCORR
    HSV[:,:,2] = HSV[:,:,2]/np.percentile(HSV[:,:,2][HSV[:,:,2]>0.], 99)
    HSV[:,:,2] = np.clip(HSV[:,:,2], 0, 1)

    print(HSV[:,:,2].min(), HSV[:,:,2].max())

    # convert back to RGB
    IMG_corr   = color.hsv2rgb(HSV)

    return IMG_corr


def scorr_fft(IMG, fname, radius=4., colorspace='hsv', trim_rad=0.7, trim_threshold=0.95):
    '''
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
    '''
    from scipy.ndimage import center_of_mass


    scale = IMG.max()

    if colorspace=='hsv':
        data  = color.rgb2hsv(IMG/scale)
        axis = 2
        invfunc = color.hsv2rgb
    elif colorspace=='lab':
        data    = color.rgb2lab(IMG/scale).astype(float)
        axis    = 0
        invfunc = color.lab2rgb
    elif colorspace=='yuv':
        data  = color.rgb2yuv(IMG/scale)
        axis = 0
        invfunc = color.yuv2rgb
    elif colorspace=='rgb':
        data  = IMG/IMG.max()
        axis  = 1
        invfunc = lambda img: img

    value = data[:,:,axis]

    # Center the image (better for the FFT)
    com    = center_of_mass(value)
    dshift = (int(IMG.shape[0]//2-com[0]), int(IMG.shape[1]//2-com[1]))

    data  = np.roll(data, dshift, axis=(0,1))
    value = data[:,:,axis]

    # Do the FFT and get the filter
    ifft2 = get_fft(value, radius=radius)
    plt.imsave(MASK_FOLDER+fname+"ifft.png", ifft2)

    # Divide the image by the filter to remove high frequency noise
    if colorspace != 'rgb':
        valnew = value/ifft2
        data[:,:,axis] = valnew
    else:
        for ci in range(3):
            data[:,:,ci] = data[:,:,ci]/ifft2

        data = data/data.max()

    data = np.roll(data, (-dshift[0], -dshift[1]), axis=(0,1))

    # Create a mask to trim the edges
    picmask = np.zeros_like(data[:,:,0])

    # Find the pixels which contain image data
    picmask[invfunc(data).min(axis=-1)>0.3] = 1.

    # Filter the mask to blur the edges
    Lfilt  = get_fft(picmask, radius=trim_rad*radius).flatten()

    # Trim the edge values based on the given threshold
    mask = Lfilt<trim_threshold

    # Save the mask as an image
    maskimg = np.zeros((data.shape[0]*data.shape[1]))
    maskimg[mask] = 1
    plt.imsave(MASK_FOLDER+"%s_Lmask.png"%fname, maskimg.reshape((IMG.shape[0], IMG.shape[1])))

    # Trim the input image with this mask
    IMGf  = data.reshape((data.shape[0]*data.shape[1], 3))
    IMGf[mask,:] = 0.
    datanew  = IMGf.reshape(IMG.shape)
    
    # Obtain the new image and transform it back to the original axis
    IMG = invfunc(datanew)*scale
    return IMG

def get_fft(value, radius):
    '''
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
    '''
    from scipy import fftpack
    ## create the low pass filter
    xx = np.asarray(range(value.shape[1]))
    yy = np.asarray(range(value.shape[0]))
    XX, YY = np.meshgrid(xx, yy)
    
    dist = (XX - xx.mean())**2. + (YY - yy.mean())**2.
    ## exponential profile
    lp_filter = np.exp(-dist/(radius**2.))

    ## apply FFT to the value component
    f1  = fftpack.fftshift(fftpack.fft2(value))
    #f1  = fftpack.fftshift(fftpack.fft2(IMGr[:,:,ci]))


    filt = np.multiply(f1, lp_filter)

    ## transform back to coordinate space
    ifft1 = np.real(fftpack.ifft2(fftpack.ifftshift(filt)))

    ## limit the filter so we don't have unwanted growth
    ifft2 = np.clip(ifft1, 0.05, 1)

    return ifft2

def box_average(IMGs, INCDs, incem, Ls, ave_all, num_procs=1):
    global shared_IMGs, shared_Ls, shared_INCDs, shared_incem, shared_IMG, \
        nfiles, nlon, nlat, ave_Ls
    nx = int(np.ceil(IMGs.shape[2]/BOX_X))
    ny = int(np.ceil(IMGs.shape[1]/BOX_Y))


    inpargs = []
    
    nfiles = IMGs.shape[0]
    nlat   = IMGs.shape[1]
    nlon   = IMGs.shape[2]
    ave_Ls = ave_all
    
    ## build the inputs to the multiprocessing pipeline
    ## this will decompose the imgs into (BOX_Y, BOX_X) slices
    for jj in range(ny):
        starty = jj*BOX_Y
        endy   = min([nlat, (jj+1)*BOX_Y])
        for ii in range(nx):
            startx  = ii*BOX_X
            endx    = min([nlon, (ii+1)*BOX_X])
           
            #inpargs.append([startxind,endxind,startyind,endyind, maski])
            inpargs.append([startx,endx,starty,endy])
    print(len(inpargs))
    ## create the final image array
    ## this will be stored as a shared array so each process 
    ## can write to it
    t0 = time.process_time()
    IMGs_ctypes  = np.ctypeslib.as_ctypes(IMGs)
    shared_IMGs  = sct.RawArray(IMGs_ctypes._type_, IMGs_ctypes)
    Ls_ctypes    = np.ctypeslib.as_ctypes(Ls)
    shared_Ls    = sct.RawArray(Ls_ctypes._type_, Ls_ctypes)
    INCDs_ctypes = np.ctypeslib.as_ctypes(INCDs)
    shared_INCDs = sct.RawArray(INCDs_ctypes._type_, INCDs_ctypes)
    incem_ctypes = np.ctypeslib.as_ctypes(incem)
    shared_incem = sct.RawArray(incem_ctypes._type_, incem_ctypes)
    
    cdtype = np.ctypeslib.as_ctypes_type(np.dtype(float))
    shared_IMG = multiprocessing.RawArray(cdtype, nlat*nlon*3)
    print(time.process_time() - t0, flush=True)

    ## start the pool
    pool = multiprocessing.Pool(processes=num_procs, initializer=initializer)
    try:
        ## start the multicore grid processing
        r = pool.map_async(do_average_box, inpargs)
        pool.close()

        tasks = pool._cache[r._job]
        ninpt = len(inpargs)
        last_update = tasks._number_left
        print("starting...")
        while tasks._number_left > 0:
            progress = (ninpt - tasks._number_left*tasks._chunksize)/ninpt
            # print a progress when the number of tasks has been updated
            if progress!=last_update:
                print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
                sys.stdout.flush()
                last_update = progress
            time.sleep(0.05)
    except Exception as e:
        pool.terminate()
        pool.join()
        raise e
        sys.exit()

    IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat,nlon, 3))
    IMG[np.isnan(IMG)] = 0.
    return IMG

def do_average_box(inp):
    global shared_IMGs, shared_Ls, shared_INCDs, shared_incem, shared_IMG, \
        nfiles, nlon, nlat, ave_Ls

    startx, endx, starty, endy = inp

    IMGs   = np.asarray(shared_IMGs, dtype=np.float32).reshape((nfiles, nlat, nlon, 3))
    INCDs  = np.asarray(shared_INCDs, dtype=np.float32).reshape((nfiles, nlat, nlon))
    Ls     = np.asarray(shared_Ls, dtype=np.float32).reshape((nfiles, nlat, nlon))
    incem  = np.asarray(shared_incem, dtype=np.float32).reshape((nfiles, nlat, nlon))
    
    IMG = np.frombuffer(shared_IMG, dtype=float).reshape((nlat, nlon, 3))

    Lsmax = np.max(Ls, axis=(1,2))

    ave_all = ave_Ls

    try:

        alpha = np.zeros((nfiles, BOX_Y, BOX_X))
        # get the images in this box
        incem_ij = incem[:,starty:endy,startx:endx]
        incds_ij = INCDs[:,starty:endy,startx:endx]
        Ls_ij    = Ls[:,starty:endy,startx:endx]
        imgs_ij  = IMGs[:,starty:endy,startx:endx]
        
        # check if there any any images in this box
        mask = ~np.isnan(incem_ij)&(Ls_ij > 0.2*Ls_ij.max())&\
            (incds_ij<np.radians(80))

        if np.sum(mask) > 1:
            # if there are, then assign weights (alpha) to each pixel in 
            # each image. final image is a linear combination of images
            # with alphas
            for kk in range(IMGs.shape[0]):
                mask = ~np.isnan(incem_ij[kk,:])&(Ls_ij[kk,:] > 0.2*Lsmax[kk])&\
                    (incds_ij[kk,:]<np.radians(80))

                # weight each image by its relative brightness wrt to the 
                # global mean
                if len(Ls_ij[kk,:,:][mask]) > 0:
                    alpha[kk,:,:][mask]  = Ls_ij[kk,:,:][mask].mean()/ave_all
                    alpha[kk,:][alpha[kk,:]!=0] = 1./alpha[kk,:][alpha[kk,:]!=0.]

            for c in range(3):
                IMGs_sub = imgs_ij[:,:,:,c]*alpha
                alpha_sum = np.sum(alpha, axis=0)
                alpha_sum[alpha_sum==0] = np.nan
                IMG[starty:endy,startx:endx,c] = np.divide(np.sum(IMGs_sub, axis=0), alpha_sum)
    except Exception as e:
        raise e

    return 0
