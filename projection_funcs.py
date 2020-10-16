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
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice
import netCDF4 as nc
import json, glob, re, os, sys
import multiprocessing, time
from scipy.interpolate import interp2d, griddata
import ctypes, time, signal

## load the C library to get the projection mask
project_c = np.ctypeslib.load_library('project.so', os.path.dirname(__file__))

image_mask_c = project_c.get_image_mask

array_1d_int    = np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS')
array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
array_2d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')

image_mask_c.argtypes = [array_1d_double, array_1d_double, ctypes.c_int, ctypes.c_int,\
                         array_2d_double, array_2d_double, array_1d_double, ctypes.c_int, ctypes.c_int]
image_mask_c.restype  = array_1d_int

process_c    = project_c.process
process_c.argtypes = [ctypes.c_double, ctypes.c_int, array_1d_double, \
                      array_2d_double, array_2d_double, array_2d_double]

## and the spice furnish function for the library
furnish_c    = project_c.furnish
furnish_c.argtypes = [ctypes.c_char_p]

KERNEL_DATAFOLDER = "/home/local/Isis/data/juno/kernels/"

FRAME_HEIGHT = 128
FRAME_WIDTH  = 1648

## filter ids for B, G and R 
FILTERS     = ['B','G','R']
CAMERA_IDS  = [-61501, -61502, -61503]

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

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def decompand(image):
    data = np.array(255.*image, dtype=np.double)
    ny, nx = data.shape

    data2 = data.copy()
    for j in range(ny):
        for i in range(nx):
            data2[j,i] = SQROOT[int(round(data[j,i]))]

    return data2

class Projector():
    def __init__(self, imagefolder, meta):
        metafile      = open(meta, 'r')
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
        
        self.load_kernels()
        
        self.re, _, self.rp = spice.bodvar(spice.bodn2c('JUPITER'), 'RADII', 3)
        self.flattening = (self.re - self.rp)/self.re

        ## calculate the start time 
        self.start_et    = spice.str2et(self.start_utc)

        self.savefolder = "%s_proj/"%self.fname

        metafile.close()

    
    def load_kernels(self):
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

        if(nck*nspk == 0):
            print("ERROR: Kernels not found for the date range!")

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
            solar_corr = np.ones((FRAME_HEIGHT, FRAME_WIDTH))
            
            '''
            for y in range(FRAME_HEIGHT):
                for x in range(23, FRAME_WIDTH-17):
                    res = \
                        self.project(x, y, cami, cam2jup, eti)
                    solar_corr[y,x] = res[3]
                    lons[y,x]       = res[4]
                    lats[y,x]       = res[5]
            '''
            process_c(eti, ci, cam2jup.flatten(), lons, lats, solar_corr)

            frame = decompand(frame[:])*solar_corr[:]

            ''' 
                find the resolution for each pixel and then calculate
                the finest resolution of the slice
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

            return (lats, lons, frame, scloc, eti, pixres)
        except:
            return

    
    def process(self, num_procs=1):
        print("%s"%self.fname)
        r = []

        done = np.zeros((self.nframes, 3))
        extents = []
        print("Projecting framelets:")
        
        lat       = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        lon       = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        decompimg = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        rawimg    = np.zeros((self.nframes, 3, FRAME_HEIGHT, FRAME_WIDTH))
        scloc     = np.zeros((self.nframes, 3))
        et        = np.zeros(self.nframes)

        inpargs = []
        for i in range(self.nframes):
            for j in range(3):
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
                print("\r[%-20s] %.2f%%"%(int(progress*20)*'=', progress*100.), end='')
                time.sleep(0.05)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit()
        
        print()
        pool.join()

        results = r.get()
        for jj in range(len(inpargs)):
            lati, loni, frame, scloci, eti, pixres[jj] \
                = results[jj]
            i, ci = inpargs[jj]
            startrow = 3*FRAME_HEIGHT*i + ci*FRAME_HEIGHT
            endrow   = 3*FRAME_HEIGHT*i +(ci+1)*FRAME_HEIGHT

            lat[i,2-ci,:,:] = lati
            lon[i,2-ci,:,:] = loni
            decompimg[i,2-ci,:,:] = frame#*scorri
            rawimg[i,2-ci,:,:]    = self.fullimg[startrow:endrow,:]
            scloc[i,:] = scloci
            et[i]      = eti
        '''
        for jj in range(len(inpargs)):
            t0 = time.perf_counter()
            lati, loni, frame, scloci, eti, pixres[jj] \
                = self.process_n_c(inpargs[jj])
            print(time.perf_counter() - t0)
            i, ci = inpargs[jj]
            startrow = 3*FRAME_HEIGHT*i + ci*FRAME_HEIGHT
            endrow   = 3*FRAME_HEIGHT*i +(ci+1)*FRAME_HEIGHT

            lat[i,2-ci,:,:] = lati
            lon[i,2-ci,:,:] = loni
            decompimg[i,2-ci,:,:] = frame#*scorri
            rawimg[i,2-ci,:,:]    = self.fullimg[startrow:endrow,:]

            scloc[i,:] = scloci
            et[i]      = eti
        '''
        pixres = pixres[pixres > 0.]

        ## save these parameters to a NetCDF file so that we can plot it later 
        f = nc.Dataset('%s.nc'%(self.fname), 'w')

        framedim = f.createDimension('nframes', self.nframes)
        coldim   = f.createDimension('ncolors', 3)
        xdim     = f.createDimension('x',FRAME_WIDTH)
        ydim     = f.createDimension('y',FRAME_HEIGHT)
        xyzdim   = f.createDimension('xyz', 3)

        ## create the NetCDF variables 
        latVar     = f.createVariable('lat', 'float32', ('nframes', 'ncolors', 'y','x'))
        lonVar     = f.createVariable('lon', 'float32', ('nframes', 'ncolors', 'y','x'))
        imgVar     = f.createVariable('img', 'float64', ('nframes', 'ncolors', 'y','x'))
        rawimgVar  = f.createVariable('rawimg', 'float64', ('nframes', 'ncolors', 'y','x'))
        scVar      = f.createVariable('scloc', 'float64', ('nframes','xyz'))
        etVar      = f.createVariable('et', 'float64', ('nframes'))

        latVar[:]    = lat[:]
        lonVar[:]    = lon[:]
        imgVar[:]    = decompimg[:]
        rawimgVar[:] = rawimg[:]
        scVar[:]     = scloc[:]
        etVar[:]     = et[:]
        
        f.close()

        mask = (lat!=-1000.)&(lon!=-1000.)
        self.lonmin = lon[mask].min()
        self.lonmax = lon[mask].max()
        self.latmin = lat[mask].min()
        self.latmax = lat[mask].max()

        print("Extents - lon: %.3f %.3f lat: %.3f %.3f - lowest pixres: %.3f deg/pix"%(\
                self.lonmin, self.lonmax, self.latmin, self.latmax, np.min(pixres)))

    def project(self, x, y, cam, cam2jup, eti):
        '''
            projects a single pixel in the filter given by 
            the cam object for a given spacecraft location
            and et
        '''
        xyvec = cam.pix2vec([x,y]).reshape(3)

        ## get the vector in the Jupiter frame
        pos_jup = np.matmul(cam2jup, xyvec)

        if((x==50)&(y==50)):
            print(xyvec)
        try:           
            point, _, srfvec = spice.sincpt("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", "JUNO", "IAU_JUPITER", pos_jup)
        except: 
            return (0., 0., 0., 1., -1000., -1000.)
        alti, loni, lati = spice.reclat(point)
        
        dist = np.linalg.norm(srfvec)/1000.

        #loni, lati, alti = spice.recpgr("JUPITER", point, self.re, self.flattening)
    
        _, _, phase, inc, emiss = spice.ilumin("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", "JUNO", point)

        mu0 = np.cos(inc)
        mu  = np.cos(emiss)
        ## Lambertian
        #solar_corr = 1./mu0
        
        ## Minnaert k=1.1
        #k = 1.1
        #solar_corr = 1./((mu0**k)*(mu**(k-1)))

        ## Lommel-Seeliger
        solar_corr = (2.*mu0)/(mu + mu0)#/(2.*mu0)

        ## Area photometric function
        #solar_corr = 2./(1. + np.cos(phase))

        '''
        if(inc < np.pi/2.):
            solar_corr = 1./np.cos(inc)
        else:
            solar_corr = 1.
        '''

        return (phase, inc, emiss, solar_corr, np.degrees(loni), np.degrees(lati))
    
class CameraModel():
    '''
        holds the camera model and filter specific
        variables
    '''
    def __init__(self, filt):
        self.filter  = filt
        self.id      = CAMERA_IDS[filt]

        ## get the camera distortion data 
        self.k1      = spice.gdpool('INS%s_DISTORTION_K1'%(self.id),0,32)[0]
        self.k2      = spice.gdpool('INS%s_DISTORTION_K2'%(self.id),0,32)[0]
        self.cx      = spice.gdpool('INS%s_DISTORTION_X'%( self.id),0,32)[0]
        self.cy      = spice.gdpool('INS%s_DISTORTION_Y'%( self.id),0,32)[0]
        self.flength = spice.gdpool('INS%s_FOCAL_LENGTH'%( self.id),0,32)[0]
        self.psize   = spice.gdpool('INS%s_PIXEL_SIZE'%(   self.id),0,32)[0]
        self.f1 = self.flength/self.psize

        ## get the timing bias 
        self.time_bias    = spice.gdpool('INS%s_START_TIME_BIAS'%self.id, 0,32)[0]
        self.iframe_delay = spice.gdpool('INS%s_INTERFRAME_DELTA'%self.id,0,32)[0]

    ''' 
    functions to obtain positions in JUNOCAM frame 
    see: https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/ik/juno_junocam_v03.ti
    '''
    def pix2vec(self, px):
        camx = px[0] - self.cx
        camy = px[1] - self.cy
        cam = self.undistort([camx, camy])
        v   = np.asarray([cam[0], cam[1], self.f1])
        return v

    def undistort(self, c):
        xd, yd = c[0], c[1]
        for i in range(5):
            r2 = (xd**2. + yd**2.)
            dr = 1. + self.k1*r2 + self.k2*r2*r2
            xd = c[0]/dr
            yd = c[1]/dr
        return (xd, yd)

    def distort(self, c):
        xd, yd = c[0], c[1]
        r2 = (xd**2+yd**2)
        dr = 1+self.k1*r2+self.k2*r2*r2
        xd *= dr
        yd *= dr
        return [xd, yd]

    def vec2pix(self, v):
        alpha = v[2]/self.f1
        cam   = [v[0]/alpha, v[1]/alpha]
        cam   = self.distort(cam)
        x     = cam[0] + self.cx
        y     = cam[1] + self.cy
        return (x,y)

def map_project_multi(files, pixres=1./25.):
    nfiles = len(files)

    lats = []
    lons = []
    for i, file in enumerate(files):
        dataset = nc.Dataset(file, 'r')

        lati = dataset.variables['lat'][:]
        loni = dataset.variables['lon'][:]
        lats.append(lati)
        lons.append(loni)


    latmin = np.min([lati[lati!=-1000].min() for lati in lats])
    latmax = np.max([lati[lati!=-1000].max() for lati in lats])
    lonmin = np.min([loni[loni!=-1000].min() for loni in lons])
    lonmax = np.max([loni[loni!=-1000].max() for loni in lons])

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
        IMGi, mask = map_project(newlon, newlat, file, True, False)

        #NPIX[:] = NPIX[:] + mask[:]
        #IMG[:]  = IMG[:] + IMGi[:]
        IMG[:]   = np.max([IMG, IMGi], axis=0)

    plt.imsave("npix.png", NPIX)

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
    IMG = IMG/(np.percentile(IMG[IMG>0.], 99.))

    plt.imsave('mosaic_RGB.png', IMG, origin='lower')
    return (newlon, newlat, IMG)

def map_project(newlon, newlat, file, save=False, savemask=False):
    fname   = file[:-3]
    print("Projecting %s"%fname)

    dataset = nc.Dataset(file, 'r')
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    imgs = dataset.variables['img'][:]
    
    scloci = dataset.variables['scloc'][:]
    eti    = dataset.variables['et'][:]

    nframes = eti.size

    jup2cam = np.zeros((nframes, 9))

    for j in range(nframes):
        jup2cam[j,:] = \
            spice.pxform("IAU_JUPITER", "JUNO_JUNOCAM", eti[j]).flatten()

    nlat = newlat.size
    nlon = newlon.size

    IMG  = np.zeros((nlat, nlon, 3))
    mask = np.zeros((nlat, nlon, 3))

    LAT, LON = np.meshgrid(newlat, newlon)

    for ci in range(3):
        print("Processing %s"%(FILTERS[2-ci]))
        lati = lats[:,ci,:,:].flatten()
        loni = lons[:,ci,:,:].flatten()
        imgi = imgs[:,ci,:,:].flatten()

        invmask = np.where((lati==-1000.)|(loni==-1000.))[0]
        ## remove pixels that were not projected
        lat = np.delete(lati, invmask)
        lon = np.delete(loni, invmask)
        img = np.delete(imgi, invmask)

        output = image_mask_c(np.radians(newlat), np.radians(newlon), nlat, nlon, \
                           scloci, jup2cam, eti, nframes, 2-ci)
        maski = ctypes.cast(output, ctypes.POINTER(ctypes.c_int*(nlat*nlon))).contents
        maski = np.asarray(maski, dtype=np.int).reshape((nlat, nlon))

        IMGI = griddata((lon, lat), img, (LON, LAT), method='cubic').T
        
        IMGI[np.isnan(IMGI)]  = 0.
        IMGI[IMGI<0.] = 0.

        print(IMGI.min(), IMGI.max())

        maski[IMGI<0.001] = 0
        ## save the mask and the raw pixel values
        if(savemask):
            plt.imsave('mask_%s_%s.png'%(fname, FILTERS[2-ci]), maski, cmap='gray', origin='lower')

        IMG[:,:,ci]  = IMGI
        mask[:,:,ci] = maski
        
        if(savemask):
            plt.imsave("%s_%s.png"%(fname, FILTERS[2-ci]), IMGI, cmap='gray', origin='lower')

    stackmask = np.min(mask,axis=2)
    for ci in range(3):
        IMG[:,:,ci] = stackmask*IMG[:,:,ci]
        mask[:,:,ci] = stackmask

    ## cleanup and do color correction
    IMG[:,:,0] *= 0.902
    IMG[:,:,2] *= 1.8879

    ## saveout here to a mosaic
    if(save):
        ## save these parameters to a NetCDF file so that we can plot it later
        f = nc.Dataset('%s_proj.nc'%fname, 'w')

        xdim     = f.createDimension('x',nlon)
        ydim     = f.createDimension('y',nlat)
        colors   = f.createDimension('colors',3)

        ## create the NetCDF variables
        latVar  = f.createVariable('lat', 'float64', ('y'))
        lonVar  = f.createVariable('lon', 'float64', ('x'))
        imgVar  = f.createVariable('img', 'float64', ('y','x','colors'))

        latVar[:]  = newlat[:]
        lonVar[:]  = newlon[:]
        imgVar[:]  = IMG[:]

        f.close()
        
        ## normalize the image by the 95% percentile
        IMG2 = IMG/(np.percentile(IMG[IMG>0.], 99.))
        plt.imsave('%s_mosaic_RGB.png'%fname, IMG2, origin='lower')
    
    return (IMG, mask)
