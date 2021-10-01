import json, glob, re
import time
from .mosaic_funcs import *

## for decompanding -- taken from Kevin Gill's github page 
SQROOT = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                   41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 67, 71, 75,
                   79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131,
                   135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
                   183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227,
                   231, 235, 239, 243, 247, 255, 263, 271, 279, 287, 295, 303,
                   311, 319, 327, 335, 343, 351, 359, 367, 375, 383, 391, 399,
                   407, 415, 423, 431, 439, 447, 455, 463, 471, 479, 487, 495,
                   503, 511, 519, 527, 535, 543, 551, 559, 567, 575, 583, 591,
                   599, 607, 615, 623, 631, 639, 647, 655, 663, 671, 679, 687,
                   695, 703, 711, 719, 727, 735, 743, 751, 759, 767, 775, 783,
                   791, 799, 807, 815, 823, 831, 839, 847, 855, 863, 871, 879,
                   887, 895, 903, 911, 919, 927, 935, 943, 951, 959, 967, 975,
                   983, 991, 999, 1007, 1023, 1039, 1055, 1071, 1087, 1103, 1119,
                   1135, 1151, 1167, 1183, 1199, 1215, 1231, 1247, 1263, 1279,
                   1295, 1311, 1327, 1343, 1359, 1375, 1391, 1407, 1439, 1471,
                   1503, 1535, 1567, 1599, 1631, 1663, 1695, 1727, 1759, 1791,
                   1823, 1855, 1887, 1919, 1951, 1983, 2015, 2047, 2079, 2111,
                   2143, 2175, 2207, 2239, 2271, 2303, 2335, 2367, 2399, 2431,
                   2463, 2495, 2527, 2559, 2591, 2623, 2655, 2687, 2719, 2751,
                   2783, 2815, 2847, 2879), dtype=np.double)


def decompand(image):
    '''
        Decompands the image from the 8-bit in the public release
        to the original 12-bit shot by JunoCam

        Parameters
        ----------
        image : numpy.ndarray
            8-bit input image

        Outputs
        -------
        data : numpy.ndarray
            Original 12-bit image
    '''
    data = np.array(255.*image, dtype=float)
    ny, nx = data.shape

    data2 = data.copy()
    for j in range(ny):
        for i in range(nx):
            data2[j,i] = SQROOT[int(round(data[j,i]))]

    return data2

class Projector():
    '''
        The main projector class that determines the surface intercept points
        of each pixel in a JunoCam image

        Methods
        -------
        load_kernels: Determines and loads the required SPICE kernels for processing
        process_n_c : Projects an individual framelet in the JunoCam raw image
        process     : Driver for the projection that handles parallel processing
    '''
    def __init__(self, imagefolder, meta, kerneldir='.'):
        with open(meta, 'r') as metafile:
            self.metadata = json.load(metafile)

        self.start_utc = self.metadata['START_TIME']
        self.fname     = self.metadata['FILE_NAME'].replace('-raw.png','')
        intframe_delay = self.metadata['INTERFRAME_DELAY'].split(' ')
        
        self.fullimg  = plt.imread(imagefolder+"%s-raw.png"%self.fname)

        self.sclat = float(self.metadata['SUB_SPACECRAFT_LATITUDE'])
        self.sclon = float(self.metadata['SUB_SPACECRAFT_LONGITUDE'])

        self.frame_delay = float(intframe_delay[0])

        ## number of strips 
        self.nframelets  = int(self.metadata['LINES']/FRAME_HEIGHT)
        ## number of RGB frames 
        self.nframes     = int(self.nframelets/3)
        
        self.load_kernels(kerneldir)
        
        self.re, _, self.rp = spice.bodvar(spice.bodn2c('JUPITER'), 'RADII', 3)
        self.flattening = (self.re - self.rp)/self.re

        ## calculate the start time 
        self.start_et    = spice.str2et(self.start_utc)

        self.savefolder = "%s_proj/"%self.fname


    
    def load_kernels(self, KERNEL_DATAFOLDER):
        '''
            Determines and loads the required SPICE data kernels
            to project and process the current image

            Parameters
            ----------
            KERNEL_DATAFOLDER : string
                path to the location of the Juno kernels

            Raises
            ------
            AssertionError
                if no kernels were found for the date range
        '''
        ## find and load the kernels for a specific date 
        iks   = sorted(glob.glob(KERNEL_DATAFOLDER+"ik/juno_junocam_v*.ti"))
        cks   = sorted(glob.glob(KERNEL_DATAFOLDER+"ck/juno_sc_rec_*.bc"))
        spks1 = sorted(glob.glob(KERNEL_DATAFOLDER+"spk/spk_rec_*.bsp"))
        spks2 = sorted(glob.glob(KERNEL_DATAFOLDER+"spk/jup*.bsp"))
        spks3 = sorted(glob.glob(KERNEL_DATAFOLDER+"spk/de*.bsp"))
        pcks  = sorted(glob.glob(KERNEL_DATAFOLDER+"pck/pck*.tpc"))
        fks   = sorted(glob.glob(KERNEL_DATAFOLDER+"fk/juno_v*.tf"))
        sclks = sorted(glob.glob(KERNEL_DATAFOLDER+"sclk/JNO_SCLKSCET.*.tsc"))
        lsks  = sorted(glob.glob(KERNEL_DATAFOLDER+"lsk/naif*.tls"))

        year, month, day = self.start_utc.split('-')
        yy = year[2:]
        mm = month
        dd = day[:2]

        intdate = int("%s%s%s"%(yy,mm,dd))

        kernels = []

        ## find the ck and spk kernels for the given date 
        ckpattern = r'juno_sc_rec_([0-9]{6})_([0-9]{6})\S*'
        nck = 0
        for ck in cks:
            fname = os.path.basename(ck)
            groups = re.findall(ckpattern, fname)
            if(len(groups) == 0):
                continue
            datestart, dateend = groups[0]

            if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                kernels.append(ck)
                nck += 1
        
        ''' use the predicted kernels if there are no rec '''
        if(nck == 0):
            ckpattern = r'juno_sc_pre_([0-9]{6})_([0-9]{6})\S*'
            for ck in cks:
                fname = os.path.basename(ck)
                groups = re.findall(ckpattern, fname)
                if(len(groups) == 0):
                    continue
                datestart, dateend = groups[0]

                if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                    kernels.append(ck)
                    nck += 1

        spkpattern = r'spk_rec_([0-9]{6})_([0-9]{6})\S*'
        nspk = 0
        for spk in spks1:
            fname = os.path.basename(spk)
            groups = re.findall(spkpattern, fname)
            if(len(groups) == 0):
                continue
            datestart, dateend = groups[0]

            if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                kernels.append(spk)
                nspk += 1

        ''' use the predicted kernels if there are no rec '''
        if(nspk == 0):
            spkpattern = r'spk_pre_([0-9]{6})_([0-9]{6})\S*'
            for spk in spks1:
                fname = os.path.basename(spk)
                groups = re.findall(spkpattern, fname)
                if(len(groups) == 0):
                    continue
                datestart, dateend = groups[0]

                if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                    kernels.append(spk)
                    nspk += 1

        #if(nck*nspk == 0):
        #    print("ERROR: Kernels not found for the date range!")
        assert nck*nspk > 0, "Kernels not found for the given date range!"

        ## load the latest updates for these 
        kernels.append(iks[-1])
        kernels.append(spks2[-1])
        kernels.append(spks3[-1])
        kernels.append(pcks[-1])
        kernels.append(fks[-1])
        kernels.append(sclks[-1])
        kernels.append(lsks[-1])

        self.kernels = kernels
        for kernel in self.kernels:
            furnish_c(kernel.encode('ascii'))
            spice.furnsh(kernel)

    def process_n_c(self, inp):
        '''
            Project a given frame and filter
            used in the multi-core version

            Parameters
            ----------
            inp : tuple
                framelet number and color index

            Outputs
            -------
            lats : numpy.ndarray
                array of latitudes in the same shape as the framelet (128,1648)
            lons : numpy.ndarray
                array of longitudes in the same shape as the framelet (128,1648)
            scloc : numpy.ndarray
                spacecraft position during the frame -- shape (3,)
            eti : double
                epoch of the spacecraft's observation for the given frame
            pixres : numpy.ndarray
                array of spatial resolutions of that pixel of shape (128,1648)
        '''
        n, ci = inp
        try:
            self.latmin =  1000.
            self.latmax = -1000.
            self.lonmin =  1000.
            self.lonmax = -1000.

            cami  = CameraModel(ci)
            start = 3*FRAME_HEIGHT*n+ci*FRAME_HEIGHT
            end   = 3*FRAME_HEIGHT*n+(ci+1)*FRAME_HEIGHT
            frame = self.fullimg[start:end,:]
            eti   = self.start_et + cami.time_bias + \
                (self.frame_delay+cami.iframe_delay)*n
            '''
                calculate the spacecraft position in the 
                Jupiter reference frame
            '''
            state, _ = spice.spkezr('JUNO', eti, 'IAU_JUPITER', 'CN', 'JUPITER')
            scloc    = state[:3]

            '''
                calculate the transformation from instrument 
                to jupiter barycenter
            '''
            cam2jup = spice.pxform('JUNO_JUNOCAM', 'IAU_JUPITER', eti)
            
            lats = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            lons = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            #solar_corr = np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            incl  =  1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            emis     =  1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            flux_cal =  np.zeros((FRAME_HEIGHT, FRAME_WIDTH))
            
            process_c(eti, ci, cam2jup.flatten(), lons, lats, incl, emis, flux_cal)

            #frame = decompand(frame[:])#*solar_corr[:]
            ''' 
                find the resolution for each pixel and then calculate
                the finest resolution of the slice
            '''

            '''
            dlats = np.gradient(lats)
            dlons = np.gradient(lons)
            
            dlat = (dlats[0] + dlats[1])/2.
            dlon = (dlons[0] + dlons[1])/2.
            dpix = np.sqrt(dlat**2. + dlon**2.)

            if(np.max(dpix) == 0.):
                pixres = 0.
            else:
                pixres = dpix[dpix>0.].min()
            '''
            pixres = 1.
            return (lats, lons, scloc, eti, pixres, incl, emis, flux_cal)
        except Exception as e:
            raise(e)
            return

    
    def process(self, num_procs=1):
        '''
            Main driver for the projection. Determines line of sight
            intercepts for each pixel on the JunoCam image

            Parameters
            ----------
            num_procs : int
                Number of CPUs to use for multiprocessing [Default: 1]
        '''
        print("%s"%self.fname)
        r = []

        done = np.zeros((self.nframes, 3))
        extents = []
        print("Projecting framelets:")
        
        lat       = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        lon       = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        decompimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        rawimg    = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        flux_cal  = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        incl      = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        emis      = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        scloc     = np.zeros((self.nframes, 3))
        et        = np.zeros(self.nframes)

        ## save these parameters to a NetCDF file so that we can plot it later 
        if not os.path.exists(NC_FOLDER):
            os.mkdir(NC_FOLDER)

        ## flatfield and gain from Brian Swift's GitHub (https://github.com/BrianSwift/JunoCam/tree/master/Juno3D)
        flatfield = np.array(io.imread(os.path.dirname(__file__)+'/cal/flatFieldsSmooth12to16.tiff'))
        #gain      = np.array(io.imread(os.path.dirname(__file__)+'/cal/gainSmooth12to16.tiff'))

        inpargs = []
        self.image = np.zeros_like(self.fullimg)
        for i in range(self.nframes):
            for j in range(3):
                startrow = 3*FRAME_HEIGHT*i + j*FRAME_HEIGHT
                endrow   = 3*FRAME_HEIGHT*i +(j+1)*FRAME_HEIGHT
                

                framei = decompand(self.fullimg[startrow:endrow,:])
                flati  = flatfield[j*FRAME_HEIGHT:(j+1)*FRAME_HEIGHT,:]
                #gaini  = gain[j*FRAME_HEIGHT:(j+1)*FRAME_HEIGHT,:]
                flati[flati==0] = 1.
                framei = framei/flati#*gaini
                self.image[startrow:endrow,:] = framei
                #framei = framei*gainSmooth12to16[j*FRAME_HEIGHT:(j+1)*FRAME_HEIGHT]

                inpargs.append((i,j))

        pixres = np.zeros(len(inpargs))
        
        pool = multiprocessing.Pool(processes=num_procs, initializer=initializer)
        try:
            r = pool.map_async(self.process_n_c, inpargs)
            pool.close()

            tasks = pool._cache[r._job]
            ninpt = len(inpargs)
            while tasks._number_left > 0:
                progress = (ninpt - tasks._number_left*tasks._chunksize)/ninpt
                print("\r[%-20s] %4.2f%%"%(int(progress*20)*'=', progress*100.), end='')
                time.sleep(0.05)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit()
        
        print()
        pool.join()

        results = r.get()
        for jj in range(len(inpargs)):
            lati, loni, scloci, eti, pixres[jj], mu0i, mui, flux_cali \
                = results[jj]
            i, ci = inpargs[jj]
            startrow = 3*FRAME_HEIGHT*i + ci*FRAME_HEIGHT
            endrow   = 3*FRAME_HEIGHT*i +(ci+1)*FRAME_HEIGHT

            lat[i,ci,:,:]       = lati
            lon[i,ci,:,:]       = loni
            decompimg[i,ci,:,:] = self.image[startrow:endrow,:]#*scorri
            rawimg[i,ci,:,:]    = self.fullimg[startrow:endrow,:]
            incl[i,ci,:,:]      = mu0i
            emis[i,ci,:,:]      = mui
            flux_cal[i,ci,:,:]  = flux_cali
            scloc[i,:]            = scloci
            et[i]                 = eti
        
        pixres = pixres[pixres > 0.]

        f = nc.Dataset('%s%s.nc'%(NC_FOLDER, self.fname), 'w')

        framedim = f.createDimension('nframes', self.nframes)
        coldim   = f.createDimension('ncolors', 3)
        xdim     = f.createDimension('x',FRAME_WIDTH)
        ydim     = f.createDimension('y',FRAME_HEIGHT)
        xyzdim   = f.createDimension('xyz', 3)

        ## create the NetCDF variables 
        latVar     = f.createVariable('lat', 'float32', ('nframes', 'ncolors', 'y','x'), zlib=True)
        lonVar     = f.createVariable('lon', 'float32', ('nframes', 'ncolors', 'y','x'), zlib=True)
        imgVar     = f.createVariable('img', 'float32', ('nframes', 'ncolors', 'y','x'), zlib=True)
        incVar     = f.createVariable('inclination', 'float32', ('nframes', 'ncolors', 'y','x'), zlib=True)
        emiVar     = f.createVariable('emission', 'float32', ('nframes', 'ncolors', 'y','x'), zlib=True)
        rawimgVar  = f.createVariable('rawimg', 'uint8', ('nframes', 'ncolors', 'y','x'), zlib=True)
        fluximgVar = f.createVariable('flux', 'float64', ('nframes', 'ncolors', 'y','x'), zlib=True)
        scVar      = f.createVariable('scloc', 'float64', ('nframes','xyz'), zlib=True)
        etVar      = f.createVariable('et', 'float64', ('nframes'), zlib=True)

        latVar[:]     = lat[:]
        lonVar[:]     = lon[:]
        imgVar[:]     = decompimg[:]
        rawimgVar[:]  = np.asarray(rawimg[:]*255,dtype=np.uint8)
        fluximgVar[:] = decompimg[:]*flux_cal
        scVar[:]      = scloc[:]
        etVar[:]      = et[:]

        incVar[:]    = incl
        emiVar[:]    = emis
        
        f.close()

        mask = (lat!=-1000.)&(lon!=-1000.)
        self.lonmin = lon[mask].min()
        self.lonmax = lon[mask].max()
        self.latmin = lat[mask].min()
        self.latmax = lat[mask].max()

        print("Extents - lon: %.3f %.3f lat: %.3f %.3f - lowest pixres: %.3f deg/pix"%(\
                self.lonmin, self.lonmax, self.latmin, self.latmax, np.min(pixres)))
    
