from .camera_funcs import *

NLAT_SLICE  = 20
NLON_SLICE  = 20

shared_lat = None
shared_lon = None
shared_img = None
shared_LON = None
shared_LAT = None
shared_IMG = None
nlon       = 0
nlat       = 0

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def map_project_multi(files, pixres=1./25., num_procs=1, extents=None, \
                      scorr_method='poly', load=False, **kwargs):
    nfiles = len(files)

    lats = []
    lons = []
    incs = []
    for i, file in enumerate(files):
        with nc.Dataset(NC_FOLDER+file, 'r') as dataset:
            lati = dataset.variables['lat'][:]
            loni = dataset.variables['lon'][:]
            inci = dataset.variables['inclination'][:]
            lats.append(lati)
            lons.append(loni)
            incs.append(inci)


    masks   = [(lats[i]!=-1000)&(lons[i]!=-1000)&(np.abs(incs[i])<np.radians(85.)) for i in range(len(files))]

    if extents is None:
        latmin = np.min([lats[i][masks[i]].min() for i in range(len(files))])
        latmax = np.max([lats[i][masks[i]].max() for i in range(len(files))])
        lonmin = np.min([lons[i][masks[i]].min() for i in range(len(files))])
        lonmax = np.max([lons[i][masks[i]].max() for i in range(len(files))])
    else:
        lonmin, lonmax, latmin, latmax = extents

    print("Extents - lon: %.3f %.3f  lat: %.3f %.3f"%(lonmin, lonmax, latmin, latmax))
    lats = None
    lons = None

    newlon = np.arange(lonmin, lonmax, pixres)
    newlat = np.arange(latmin, latmax, pixres)

    #LAT, LON = np.meshgrid(newlat, newlon)

    nlat = newlat.size
    nlon = newlon.size 
    
    IMG   = np.zeros((nlat, nlon, 3))
    IMGs  = np.zeros((len(files), nlat, nlon, 3))
    Ls    = np.zeros((len(files), nlat, nlon))
    INCLs = np.zeros((len(files), nlat, nlon))
    EMISs = np.zeros((len(files), nlat, nlon))
    masks = np.zeros((len(files), nlat, nlon))
    NPIX  = np.zeros((nlat, nlon, 3), dtype=np.int)

    print("Mosaic shape: %d x %d"%(nlon, nlat))

    for i, file in enumerate(files):
        fname = files[i][:-3]
        _, IMGi, masks[i,:], INCLs[i,:], EMISs[i,:] = map_project(file, newlon=newlon, newlat=newlat,\
                        save=True, savemask=True, num_procs=num_procs, \
                        scorr_method=scorr_method, load=load, ret_inc=True, **kwargs)

        #NPIX[:] = NPIX[:] + mask[:]
        #IMG[:]  = IMG[:] + IMGi[:]
        #IMG[:]   = np.max([IMG, IMGi], axis=0)
        IMGs[i,:] = IMGi
        Ls[i,:]   = color.rgb2hsv(IMGi)[:,:,2]

    combine_method = kwargs.get('combine_method', 'max')

    if combine_method=='min_grad':
        Lx, Ly = np.gradient(Ls, axis=(1,2))
        Lxx = np.gradient(Lx, axis=1)
        Lyy = np.gradient(Ly, axis=2)
        delsqL = np.abs(Lx + Ly)

    npix    = np.sum(Ls>0.2, axis=0)
    overlap_mask = npix>1

    ave_val = np.zeros(len(files))
    for i in range(len(files)):
        imgi = Ls[i,:][overlap_mask]
        ave_val[i] = (imgi[imgi>0.]).mean()

    ave_all = np.mean(ave_val)
    print(ave_val, ave_all)

    for i in range(len(files)):
        IMGs[i,:] *= ave_all/ave_val[i]


    if combine_method=='max':
        combine = np.max
    elif combine_method=='min':
        combine = np.min

    if 'none' in scorr_method:
        IMG = np.max(IMGs, axis=0)
        incem = None
    else:
        incem = INCLs*EMISs
        incem[IMGs.sum(axis=-1)<0.02] = np.nan
        print("Mosaicing image")
        for jj in range(IMG.shape[0]):
            print("\r[%-20s] %d/%d"%(int(jj/IMG.shape[0]*20.)*'=', jj, IMG.shape[0]), end='')
            for ii in range(IMG.shape[1]):
                incemij = incem[:,jj,ii]
                mask = ~np.isnan(incemij)&(Ls[:,jj,ii] > 0.2)&\
                    (INCLs[:,jj,ii]<np.radians(80))
                nimgs = np.sum(mask)
                if nimgs > 1:
                    if combine_method=='min_grad':
                        IMG[jj,ii,:] = IMGs[np.argmin(delsqL[mask,jj,ii]),jj,ii,:]
                    else:
                        IMG[jj,ii,:] = combine(IMGs[mask,jj,ii,:], axis=0)
                elif nimgs > 0:
                    IMG[jj,ii,:] = IMGs[mask,jj,ii,:]


    #IMG[NPIX>0] = IMG[NPIX>0]/NPIX[NPIX>0]

    ## save these parameters to a NetCDF file so that we can plot it later 
    with nc.Dataset(NC_FOLDER+'multi_proj_raw.nc', 'w') as f:
        xdim     = f.createDimension('x',nlon)
        ydim     = f.createDimension('y',nlat)
        colors   = f.createDimension('colors',3)
        files    = f.createDimension('file',len(files))

        ##  create the NetCDF variables 
        latVar   = f.createVariable('lat', 'float64', ('y'))
        lonVar   = f.createVariable('lon', 'float64', ('x'))
        imgVar   = f.createVariable('img', 'float64', ('y','x','colors'))
        imgsVar  = f.createVariable('imgs', 'float64', ('file', 'y','x','colors'))
        if incem is not None:
            incemVar = f.createVariable('incem', 'float64', ('file', 'y','x'))

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]
        imgsVar[:] = IMGs
        if incem is not None:
            incemVar[:] = incem[:]
    
    ## normalize the image by the 99% percentile 
    IMG = IMG/(np.percentile(IMG[IMG>0.], 99.))

    IMG = np.clip(IMG, 0., 1.)

    plt.imsave(MOS_FOLDER+'mosaic_RGB.png', IMG, origin='lower')
    return (newlon, newlat, IMG)

def map_project(file, newlon=None, newlat=None, pixres=None, num_procs=1, \
                save=False, savemask=False, scorr_method='simple', load=False, ret_inc=False,
                **kwargs):
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
        scorr_method : string
            The type of lighting correction to use:
            simple : assume Jupiter is a Lambertian surface (works decently
                    well for some images but does not handle terminators well)
            poly :   fit a 2nd order polynomial in mu and mu0 based on the 
                    brightness of the image 
    '''
    global shared_LON, shared_LAT, nlon, nlat 

    fname   = file[:-3]
    print("Projecting %s"%fname)

    if not os.path.exists(MOS_FOLDER):
        os.mkdir(MOS_FOLDER)
    if not os.path.exists(RGB_FOLDER):
        os.mkdir(RGB_FOLDER)
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
    incl   = dataset.variables['inclination'][:]
    emis   = dataset.variables['emission'][:]

    nframes = eti.shape[0]

    if (newlon is None) and (newlat is None) and (pixres is not None):
        masks   = (lats!=-1000)&(lons!=-1000)&(np.abs(incl)<np.radians(89.))
        latmin = lats[masks].min()
        latmax = lats[masks].max()
        lonmin = lons[masks].min()
        lonmax = lons[masks].max()
        newlon = np.arange(lonmin, lonmax, pixres)
        newlat = np.arange(latmin, latmax, pixres)
        print("Limits: lon: %.3f %.3f  lat: %.3f %.3f  size: %d x %d"%(\
                newlon.min(), newlon.max(), newlat.min(), newlat.max(), newlon.size, newlat.size))
    elif (newlon is not None) and (newlat is not None):
        pass
    else:
        raise RuntimeError("Please provide either lat/lon or resolution")

    ## define the arrays to hold the new gridded data
    nlat = newlat.size
    nlon = newlon.size
    IMG  = np.zeros((nlat, nlon, 3))
    LAT, LON = np.meshgrid(newlat, newlon)
    
    ## create a shared memory object for LON/LAT
    LON_ctypes = np.ctypeslib.as_ctypes(LON)
    shared_LON = sct.RawArray(LON_ctypes._type_, LON_ctypes)
    LAT_ctypes = np.ctypeslib.as_ctypes(LAT)
    shared_LAT = sct.RawArray(LAT_ctypes._type_, LAT_ctypes)

    ## get the image mask where no data exists
    ## this is created to remove errors from interpolation
    if load&os.path.exists(NPY_FOLDER+"%s_mask.npy"%fname):
        print("Loading mask file")
        maski = np.load(NPY_FOLDER+"%s_mask.npy"%fname)
    else:
        output = image_mask_c(np.radians(newlat), np.radians(newlon), nlat, nlon, \
                           eti, nframes)
        maski = ctypes.cast(output, ctypes.POINTER(ctypes.c_int*(nlat*nlon))).contents
        maski = np.asarray(maski, dtype=np.int).reshape((nlat, nlon))
        np.save(NPY_FOLDER + "%s_mask.npy"%fname, maski)

    ## save the mask and the raw pixel values if needed
    if(savemask):
        plt.imsave(RGB_FOLDER+'mask_%s.png'%(fname), maski, vmin=0., vmax=1., cmap='gray', origin='lower')

    for ci in range(3):
        print("Processing %s"%(FILTERS[ci]))
        filteriname = NPY_FOLDER + "%s_%s.npy"%(fname, FILTERS[ci])
        if load&os.path.exists(filteriname):
            print("Loading from %s"%filteriname)
            IMGI = np.load(filteriname)
        else:
            lati = lats[:,ci,:,:].flatten()
            loni = lons[:,ci,:,:].flatten()
            imgi = imgs[:,ci,:,:].flatten()
            emi  = emis[:,ci,:,:].flatten()
            inci = incl[:,ci,:,:].flatten()

            invmask = np.where((lati==-1000.)|(loni==-1000.)|(np.abs(inci)>np.radians(89.)))[0]
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
            if(save):
                np.save(filteriname, IMGI)
        maski[IMGI<0.001] = 0
        IMG[:,:,ci]  = IMGI

    INCL, EMIS = get_emis_incl_map(incl, emis, lats, lons, \
                                   newlat, newlon, maski, fname, \
                                   num_procs=num_procs,\
                                   load=load)

    ## apply lighting correction
    ## mask out all data with high incidence and
    ## emission angles
    #maski[np.abs(INCL)>np.radians(89)] = 0
    #maski[np.abs(EMIS)>np.radians(89)] = 0

    maski = np.clip(maski, 0, 1)
    for ci in range(3):
        IMG[:,:,ci] = maski*IMG[:,:,ci]

    ## switch from BGR to RGB
    IMG = IMG[:,:,::-1]
    
    ## do color correction
    #IMG[:,:,0] *= 0.902
    #IMG[:,:,2] *= 1.8879

    #IMG = IMG/np.percentile(IMG[IMG>0], 99)
    #IMG = np.clip(IMG, 0., 1.)#IMG/np.percentile(IMG, 99)

    ## Do a simple Lambertian correction if needed
    if 'simple' in scorr_method:
        mu0 = np.clip(np.cos(INCL), 0, 1)
        mu0[mu0<0.01] = np.nan
        scorr = mu0#2.*mu0/(mu0 + mu)
        for ci in range(3):
            IMG[:,:,ci] = IMG[:,:,ci]/scorr


    if 'poly' in scorr_method:
        IMG = scorr_poly(INCL, EMIS, newlat, newlon, IMG)

    IMG[np.isnan(IMG)] = 0.

    if 'fft' in scorr_method:
        IMG = scorr_fft(IMG, fname, \
                        radius=kwargs.get('fft_radius', 5.), \
                        colorspace=kwargs.get('colorspace', 'hsv'), \
                        trim_rad=kwargs.get('trim_rad', 0.6))
        
        ## remove the fringes (brightest and dimmest clusters from SLIC)
        #labels = slic(IMG, compactness=0.5, n_segments=500, \
                      #start_label=1, convert2lab=True)
        #IMGsp = color.label2rgb(labels, IMG/IMG.max(), kind='avg', bg_label=0)
        #Lsp   = color.rgb2lab(IMGsp)[:,:,0]

        #IMG[IMG>topp] = 0.
    


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

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]
        img_corrVar[:] = np.asarray(IMGc*255/IMG.max(), dtype=np.uint8)

    if ret_inc:
        return "%s_proj.nc"%fname, IMG, maski, INCL, EMIS
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
            print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
            time.sleep(0.05)

        # for ri in pool.imap_unordered(project_part_image, inpargs):
        #     progress = i/len(inpargs)
        #     print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
        #     time.sleep(0.05)

        #     ## construct the image array from the output
        #     # startxind, endxind, startyind, endyind, IMGi = ri
        #     # IMG[startyind:endyind,startxind:endxind] = IMGi
        #     i += 1 
        # pool.close()

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
                     trim_saturated=False, sat_threshold=95, **kwargs):
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

        if trim_saturated:
            scaling = np.percentile(IMG2[IMG2>0.], sat_threshold)
            IMGc = IMG2/scaling
            IMGc[IMGc.sum(axis=2)>2.98] = 0.
            IMG2 = IMGc*scaling

        if hist_eq:
            ## normalize the image by the 95% percentile
            IMG2 = IMG2/(np.percentile(IMG2[IMG2>0.], 99.))
            IMG2 = np.clip(IMG2, 0, 1)
            for ci in range(3):
                val = IMG2[:,:,ci]
                p0, p1 = np.percentile(val[val>0.], (2,99))
                #hsv[:,:,2] = exposure.rescale_intensity(hsv[:,:,2], in_range=(p0, p1))
                #hsv[:,:,2] = exposure.equalize_adapthist(hsv[:,:,2], clip_limit=0.05)
                IMG2[:,:,ci] = exposure.equalize_adapthist(IMG2[:,:,ci], clip_limit=kwargs.get('clip_limit', 0.05))#color.hsv2rgb(hsv)

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

def get_emis_incl_map(incl, emis, lat, lon, newlat, newlon, maski, \
                      fname, num_procs=1, load=True):
    ## flatten and mask out the bad data points
    incl = incl.flatten(); emis = emis.flatten();
    lat  = lat.flatten();  lon  = lon.flatten()
    mask = (lat!=-1000)&(lon!=-1000)&(np.abs(incl)<np.radians(89.))

    inclsf  = incl[mask].flatten()
    emisf   = emis[mask].flatten()
    lonf    = lon[mask].flatten()
    latf    = lat[mask].flatten()

    ## create the new grid to project onto and
    ## project the inclination and emission values onto the new grid
    LAT, LON = np.meshgrid(newlat, newlon)
    #EMIS = griddata((lonf, latf), emisf, (LON, LAT)).T
    #INCL = griddata((lonf, latf), inclsf, (LON, LAT)).T

    if load&os.path.exists(NPY_FOLDER+"%s_emis.npy"%fname):
        print("Loading emission data")
        EMIS = np.load(NPY_FOLDER+"%s_emis.npy"%fname)
    else:
        EMIS = project_to_uniform_grid(lonf, latf, emisf, num_procs=num_procs)
        EMIS[~maski] = np.nan
        np.save(NPY_FOLDER+"%s_emis.npy"%fname, EMIS)
    
    if load&os.path.exists(NPY_FOLDER+"%s_incl.npy"%fname):
        print("Loading inclination data")
        INCL = np.load(NPY_FOLDER+"%s_incl.npy"%fname)
    else:
        INCL = project_to_uniform_grid(lonf, latf, inclsf, num_procs=num_procs)
        INCL[~maski] = np.nan
        np.save(NPY_FOLDER+"%s_incl.npy"%fname, INCL)

    return (INCL, EMIS)

def scorr_poly(INCL, EMIS, newlat, newlon, IMG):
    '''
        Fit a 2nd order polynomial in inclination/emission space
        to correct for lighting geometry
    '''
    print("Doing polynomial lighting correction")


    mask = (~np.isnan(EMIS))&(~np.isnan(INCL))&(IMG.min(axis=2)>0.)

    ## get the brightness value of the pixels
    LAB = color.rgb2hsv(IMG/IMG.max())
    IMG_val = LAB[:,:,2]/100.
    print(IMG_val.min(), IMG_val.max())
    #IMG_val = IMG[:,:,1]

    
    INCLf = np.cos(INCL[mask].flatten())
    EMISf = np.cos(EMIS[mask].flatten())
    VALf  = IMG_val[mask].flatten()

    ## remove boundary data
    mask = (INCLf>0.05)
    INCLf = INCLf[mask]
    EMISf = EMISf[mask]
    VALf  = VALf[mask]

    ind = np.asarray(range(len(INCLf)))
    np.random.shuffle(ind)
    ind = ind[:1000]

    INCLf = INCLf[ind]; EMISf = EMISf[ind]; VALf = VALf[ind]

    ## fit the 2nd order polynomial
    x  = INCLf; y = EMISf
    xx = np.cos(INCL); yy = np.cos(EMIS)

    m = 5 ## polynomial order
    A = []
    for i in range(m):
        for j in range(i):
            A.append( (x**i)*(y**j) )
    A = np.asarray(A).T
    coeff, r, rank, s = np.linalg.lstsq(A, VALf, rcond=None)

    SCORR = np.zeros_like(INCL)
    k = 0
    for i in range(m):
        for j in range(i):
            SCORR += coeff[k]*(xx**i)*(yy**j)
            k += 1


    '''
    A = np.array([INCLf*0 + 1, INCLf, EMISf, INCLf**2., INCLf**2.*EMISf,\
                  INCLf**2.*EMISf**2., EMISf**2.*INCLf, EMISf**2., INCLf*EMISf, \
                  INCLf**3., INCLf**3.*EMISf, INCLf**3.*EMISf**2., INCLf**3.*EMISf**3,\
                  INCLf**2.*EMISf**3., INCLf*EMISf**3., EMISf**3.
                  ]).T
    coeff, r, rank, s = np.linalg.lstsq(A, VALf)

    ## create the correction image
    SCORR = coeff[0] + np.cos(INCL)*coeff[1] + np.cos(EMIS)*coeff[2] + np.cos(INCL)**2.*coeff[3] + \
        np.cos(INCL)**2.*np.cos(EMIS)*coeff[4] + np.cos(INCL)**2.*np.cos(EMIS)**2.*coeff[5] + \
        np.cos(EMIS)**2.*np.cos(INCL)*coeff[6] + np.cos(EMIS)**2.*coeff[7] + np.cos(EMIS)*np.cos(INCL)*coeff[8] + \
        np.cos(INCL)**3.*coeff[9] + np.cos(INCL)**3.*np.cos(EMIS)*coeff[10] + \
        np.cos(INCL)**3.*np.cos(EMIS)**2.*coeff[11] + np.cos(INCL)**3.*np.cos(EMIS)**3.*coeff[12] +\
        np.cos(INCL)**2.*np.cos(EMIS)**3.*coeff[13] + np.cos(INCL)*np.cos(EMIS)**3.*coeff[14] + \
        np.cos(EMIS)**3.*coeff[15]
    
    ## fit the 2nd order polynomial
    A = np.array([INCLf*0 + 1, INCLf, EMISf, INCLf**2., INCLf**2.*EMISf,\
                  INCLf**2.*EMISf**2., EMISf**2.*INCLf, EMISf**2., INCLf*EMISf]).T
    coeff, r, rank, s = np.linalg.lstsq(A, VALf)

    ## create the correction image
    SCORR = coeff[0] + np.cos(INCL)*coeff[1] + np.cos(EMIS)*coeff[2] + np.cos(INCL)**2.*coeff[3] + \
        np.cos(INCL)**2.*np.cos(EMIS)*coeff[4] + np.cos(INCL)**2.*np.cos(EMIS)**2.*coeff[5] + \
        np.cos(EMIS)**2.*np.cos(INCL)*coeff[6] + np.cos(EMIS)**2.*coeff[7] + np.cos(EMIS)*np.cos(INCL)*coeff[8]
    
    A = np.array([INCLf*0 + 1, INCLf, EMISf, INCLf*EMISf]).T
    coeff, r, rank, s = np.linalg.lstsq(A, VALf)

    ## create the correction image
    SCORR = coeff[0] + np.cos(INCL)*coeff[1] + np.cos(EMIS)*coeff[2] + \
        np.cos(EMIS)*np.cos(INCL)*coeff[3]
    '''

    SCORR[SCORR<0.] = np.nan
    #SCORR[SCORR<0.05] = 0.05

    ## correct the image and get the RGB values back
    LAB[:,:,0] = LAB[:,:,0]/SCORR
    LAB[:,:,0] = LAB[:,:,0]/np.percentile(LAB[:,:,0][LAB[:,:,0]>0.], 99)
    LAB[:,:,0] = np.clip(LAB[:,:,0], 0, 1)*100.

    print(LAB[:,:,0].min(), LAB[:,:,0].max())

    IMG_corr   = color.lab2rgb(LAB)

    

    return IMG_corr


def scorr_fft(IMG, fname, radius=4., colorspace='hsv', trim_rad=0.7):
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

    com    = center_of_mass(value)
    dshift = (int(IMG.shape[0]//2-com[0]), int(IMG.shape[1]//2-com[1]))

    data  = np.roll(data, dshift, axis=(0,1))
    value = data[:,:,axis]

    ifft2 = get_fft(value, radius=radius)
    plt.imsave(fname+"ifft.png", ifft2)

    #IMGr[:,:,ci] = IMGr[:,:,ci]/ifft2
    if colorspace != 'rgb':
        valnew = value/ifft2
        '''
        if colorspace in ['hsv', 'yuv']:
            scale = np.max(valnew)#np.median(valnew[valnew>0.])/0.4
            valnew = np.clip(valnew/scale, 0, 1)
        elif colorspace=='lab':
            scale = np.mean(valnew[valnew>0.])/40.
            valnew = np.clip(valnew/scale, 0, 100)
        '''
        data[:,:,axis] = valnew
    else:
        for ci in range(3):
            data[:,:,ci] = data[:,:,ci]/ifft2

        data = data/data.max()

    IMG = invfunc(np.roll(data, (-dshift[0], -dshift[1]), axis=(0,1)))*scale

    picmask = np.zeros_like(IMG[:,:,0])
    picmask[IMG.mean(axis=2)>0.] = 1.
    Lfilt  = get_fft(picmask, radius=trim_rad*radius).flatten()
    mask = Lfilt<0.95
    maskimg = np.zeros((IMG.shape[0]*IMG.shape[1]))
    maskimg[mask] = 1
    plt.imsave(RGB_FOLDER+"%s_Lmask.png"%fname, maskimg.reshape((IMG.shape[0], IMG.shape[1])))
    IMGf  = IMG.reshape((IMG.shape[0]*IMG.shape[1], 3))
    IMGf[mask,:] = 0.
    IMG   = IMGf.reshape(IMG.shape)

    #hsv[:,:,2] = L/L.max()
    #for ci in range(3):
    #    imgi = np.roll(IMG[:,:,ci], dshift, axis=(0,1))
    #    IMG[:,:,ci] = np.roll(imgi/ifft2, (-dshift[0], -dshift[1]), axis=(0,1))

    #return IMG#np.roll(IMG, (-dshift[0], -dshift[1]), axis=(0,1))
    return IMG

def get_fft(value, radius):
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
