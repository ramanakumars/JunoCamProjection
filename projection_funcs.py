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
import json, glob, re, os
import multiprocessing
from scipy.interpolate import interp2d, griddata

KERNEL_DATAFOLDER = "/home/local/Isis/data/juno/kernels/"

IMAGE_HEIGHT = 15360
FRAME_HEIGHT = 128
FRAME_WIDTH  = 1648

''' filter ids for B, G and R '''
FILTERS     = ['B', 'G', 'R']
CAMERA_IDS  = [-61501, -61502, -61503]

''' functions to obtain positions in JUNOCAM frame '''
def pix2vec(px, filt):
    camx = px[0] - cx[filt]
    camy = px[1] - cy[filt]
    cam = undistort([camx, camy], filt)
    v   = np.array([cam[0], cam[1], f1[filt]])

    return v

def undistort(c, filt):
    xd, yd = c[0], c[1]
    for i in range(5):
        r2 = (xd**2. + yd**2.)
        dr = 1. + k1[filt]*r2 + k2[filt]*r2*r2
        xd = c[0]/dr
        yd = c[1]/dr
    return (xd, yd)

class Projector():
    def __init__(self, image, meta):
        metafile      = open(meta, 'r')
        self.metadata = json.load(metafile)
        self.fullimg  = plt.imread(image)

        self.start_utc = self.metadata['START_TIME']
        self.fname     = self.metadata['FILE_NAME'].replace('-raw.png','')
        intframe_delay = self.metadata['INTERFRAME_DELAY'].split(' ')

        self.frame_delay = float(intframe_delay[0])

        ''' number of strips '''
        self.nframelets  = int(self.metadata['LINES']/FRAME_HEIGHT)


        ''' number of RGB frames '''
        self.nframes     = int(self.nframelets/3)
        
        self.load_kernels()

        ''' calculate the start time '''
        self.start_et    = spice.str2et(self.start_utc)

        self.savefolder = "%s_proj/"%self.fname

        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
            print("Folder was created: ", self.savefolder)
        metafile.close()
    
    def load_kernels(self):
        ''' 
        find and load the kernels for a specific date 
        '''
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
        ''' load the latest updates for these '''
        kernels.append(iks[-1])
        kernels.append(spks2[-1])
        kernels.append(spks3[-1])
        kernels.append(pcks[-1])
        kernels.append(fks[-1])
        kernels.append(sclks[-1])
        kernels.append(lsks[-1])

        ''' find the ck and spk kernels for the given date '''
        ckpattern = r'juno_sc_rec_([0-9]{6})_([0-9]{6})\S*'
        for ck in cks:
            fname = os.path.basename(ck)
            groups = re.findall(ckpattern, fname)
            if(len(groups) == 0):
                continue
            datestart, dateend = groups[0]

            if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                kernels.append(ck)
        
        spkpattern = r'spk_rec_([0-9]{6})_([0-9]{6})\S*'
        for spk in spks1:
            fname = os.path.basename(spk)
            groups = re.findall(spkpattern, fname)
            if(len(groups) == 0):
                continue
            datestart, dateend = groups[0]

            if( (int(datestart) <= intdate) & (int(dateend) >= intdate) ):
                kernels.append(spk)
        self.kernels = kernels
        for kernel in self.kernels:
            spice.furnsh(kernel)

    def process(self):
        '''
            Single core version of the projection function
        '''
        RCam = CameraModel(2)
        GCam = CameraModel(1)
        BCam = CameraModel(0)

        cams = [BCam, GCam, RCam]

        self.latmin =  1000.
        self.latmax = -1000.
        self.lonmin =  1000.
        self.lonmax = -1000.
        
        for n in range(self.nframes):
            for ci in range(3):
                print("Frame: %d %s"%(n, FILTERS[ci]))
                cami  = cams[ci]
                start = 3*FRAME_HEIGHT*n+ci*FRAME_HEIGHT
                end   = 3*FRAME_HEIGHT*n+(ci+1)*FRAME_HEIGHT
                frame = self.fullimg[start:end,:]
                
                eti   = self.start_et + cami.time_bias + \
                    (self.frame_delay+cami.iframe_delay)*n
                '''
                    calculate the spacecraft position in the 
                    Jupiter reference frame
                '''
                state, _ = spice.spkezr('JUNO', eti, 'IAU_JUPITER', 'CN+S', 'JUPITER')
                scloc    = state[:3]

                '''
                    calculate the transformation from instrument 
                    to jupiter barycenter
                '''
                cam2jup = spice.pxform('JUNO_JUNOCAM', 'IAU_JUPITER', eti)
                
                lats = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
                lons = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
                solar_corr = np.ones((FRAME_HEIGHT, FRAME_WIDTH))
                
                for y in range(FRAME_HEIGHT):
                    if(y%10==0):
                        progress = (y/FRAME_HEIGHT)
                        print("\r%d [%-20s] %.2f%%"%(n, int(progress*20)*'=', progress*100.), end='')
                    for x in range(23, FRAME_WIDTH-17):
                        solar_corr[y,x], lons[y,x], lats[y,x] = \
                            self.project(x, y, cami, scloc, cam2jup, eti)
 
                mask = lats.flatten() != -1000.
                
                if(sum(mask) != 0):
                    latmin = lats.flatten()[mask].min()
                    latmax = lats.flatten()[mask].max()
                    lonmin = lons.flatten()[mask].min()
                    lonmax = lons.flatten()[mask].max()
                    print(" Lat: {:7.3f} {:7.3f} Lon: {:7.3f} {:7.3f}".format(\
                        latmin, latmax, lonmin, lonmax))
                else:
                    latmin = self.latmin
                    latmax = self.latmax
                    lonmin = self.lonmin
                    lonmax = self.lonmax
                    print()


                ''' save these parameters to a NetCDF file so that we can plot it later '''
                f = nc.Dataset('%s/%s_proj_%s_%d.nc'%(self.savefolder,self.fname, FILTERS[ci], n), 'w')

                xdim     = f.createDimension('x',FRAME_WIDTH)
                ydim     = f.createDimension('y',FRAME_HEIGHT)

                ''' create the NetCDF variables '''
                latVar  = f.createVariable('lat', 'float64', ('y','x'))
                lonVar  = f.createVariable('lon', 'float64', ('y','x'))
                imgVar  = f.createVariable('img', 'float64', ('y','x'))

                latVar[:]  = lats[:]
                lonVar[:]  = lons[:]
                imgVar[:]  = frame[:]*solar_corr[:]

                f.close()

                self.latmin = min([latmin, self.latmin])
                self.latmax = max([latmax, self.latmax])
                self.lonmin = min([lonmin, self.lonmin])
                self.lonmax = max([lonmax, self.lonmax])

    def process_n_c(self, n, ci):
        '''
            Project a given frame and filter
            used in the multi-core version
        '''
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
        state, _ = spice.spkezr('JUNO', eti, 'IAU_JUPITER', 'CN+S', 'JUPITER')
        scloc    = state[:3]

        '''
            calculate the transformation from instrument 
            to jupiter barycenter
        '''
        cam2jup = spice.pxform('JUNO_JUNOCAM', 'IAU_JUPITER', eti)
        
        lats = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
        lons = -1000.*np.ones((FRAME_HEIGHT, FRAME_WIDTH))
        solar_corr = np.ones((FRAME_HEIGHT, FRAME_WIDTH))
        
        for y in range(FRAME_HEIGHT):
            for x in range(23, FRAME_WIDTH-17):
                solar_corr[y,x], lons[y,x], lats[y,x] = \
                    self.project(x, y, cami, scloc, cam2jup, eti)

        mask = lats.flatten() != -1000.
        
        if(sum(mask) != 0):
            latmin = lats.flatten()[mask].min()
            latmax = lats.flatten()[mask].max()
            lonmin = lons.flatten()[mask].min()
            lonmax = lons.flatten()[mask].max()
        else:
            latmin = self.latmin
            latmax = self.latmax
            lonmin = self.lonmin
            lonmax = self.lonmax

        ''' save these parameters to a NetCDF file so that we can plot it later '''
        f = nc.Dataset('%s/%s_proj_%s_%d.nc'%(self.savefolder,self.fname, FILTERS[ci], n), 'w')

        xdim     = f.createDimension('x',FRAME_WIDTH)
        ydim     = f.createDimension('y',FRAME_HEIGHT)

        ''' create the NetCDF variables '''
        latVar  = f.createVariable('lat', 'float64', ('y','x'))
        lonVar  = f.createVariable('lon', 'float64', ('y','x'))
        imgVar  = f.createVariable('img', 'float64', ('y','x'))

        latVar[:]  = lats[:]
        lonVar[:]  = lons[:]
        imgVar[:]  = frame[:]*solar_corr[:]

        f.close()

        self.latmin = min([latmin, self.latmin])
        self.latmax = max([latmax, self.latmax])
        self.lonmin = min([lonmin, self.lonmin])
        self.lonmax = max([lonmax, self.lonmax])

        return (self.lonmin, self.lonmax, self.latmin, self.latmax)

    def project(self, x, y, cam, scloc, cam2jup, eti):
        '''
            projects a single pixel in the filter given by 
            the cam object for a given spacecraft location
            and et
        '''
        xyvec = cam.pix2vec([x+0.5,y+0.5]).reshape(3)

        ## get the vector in the Jupiter frame
        pos_jup = np.matmul(cam2jup, xyvec)
        try:           
            point, _, _ = spice.sincpt("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO", "IAU_JUPITER", pos_jup)
        except: 
            return (1., -1000., -1000.)
        dist, loni, lati = spice.reclat(point)

        _, _, phase, inc, emiss = spice.ilumin("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO", point)

        solar_corr = 1./np.cos(inc)

        return (solar_corr, np.degrees(loni), np.degrees(lati))


class CameraModel():
    '''
        holds the camera model and filter specific
        variables
    '''
    def __init__(self, filt):
        self.filter  = filt
        self.id      = CAMERA_IDS[filt]


        ''' get the camera distortion data '''
        self.k1      = spice.gdpool('INS%s_DISTORTION_K1'%(self.id),0,32)[0]
        self.k2      = spice.gdpool('INS%s_DISTORTION_K2'%(self.id),0,32)[0]
        self.cx      = spice.gdpool('INS%s_DISTORTION_X'%( self.id),0,32)[0]
        self.cy      = spice.gdpool('INS%s_DISTORTION_Y'%( self.id),0,32)[0]
        self.flength = spice.gdpool('INS%s_FOCAL_LENGTH'%( self.id),0,32)[0]
        self.psize   = spice.gdpool('INS%s_PIXEL_SIZE'%(   self.id),0,32)[0]
        self.f1 = self.flength/self.psize

        ''' get the timing bias '''
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


def do_mp(image, meta, num_procs):
    '''
        do a multi-core projection of a given image
    '''
    dummy_proj = Projector(image, meta)

    r = []
    pool = multiprocessing.Pool(processes=num_procs)

    done = np.zeros((dummy_proj.nframes, 3))
    extents = []
    print("Projecting framelets:")
    try:
        for n in range(dummy_proj.nframes):
            for ci in range(3):
                def call(res, nn=n, cc=ci):
                    extents.append(res)
                    
                    done[nn,cc] = 1
                    progress = done.sum()/(done.size)
                    print("\r[%-20s] %.3f%%"%(int(progress*20)*'=', progress*100.), end='')
                projnc = Projector(image, meta)
                r.append(pool.apply_async(projnc.process_n_c, (n, ci,), callback=call))
        pool.close()
        for item in r:
            item.wait()
    except KeyboardInterrupt:
        pool.terminate()
        return
    finally:
        pool.join()

    extents = np.array(extents)
    lonmin = np.min(extents[:,0])
    lonmax = np.max(extents[:,1])
    latmin = np.min(extents[:,2])
    latmax = np.max(extents[:,3])
    print()
    print("Extents - lon: %.3f %.3f lat: %.3f %.3f"%(lonmin, lonmax, latmin, latmax))

def create_RGB_frame(folder, extents=None, pixres=1./75., padding=10):
    '''
        given a folder of map-projected data, create an RGB image
        in a lat/lon projection
    '''
    try:
        lonmin, lonmax, latmin, latmax = extents
    except:
        lonmin, lonmax, latmin, latmax = [-180., 180., -90., 90.]
    newlon = np.arange(lonmin, lonmax, pixres)
    newlat = np.arange(latmin, latmax, pixres)

    nlat = newlat.shape[0]
    nlon = newlon.shape[0]

    ''' create edges for the 2d histogram '''
    newlatbins = np.zeros(nlat+1)
    newlatbins[:-1] = newlat-pixres
    newlatbins[-1]  = newlat[-1] + pixres
    newlonbins = np.zeros(nlon+1)
    newlonbins[:-1] = newlon-pixres
    newlonbins[-1]  = newlon[-1] + pixres

    IMG  = np.zeros((nlat, nlon, 3))
    LAT, LON = np.meshgrid(newlat, newlon)

    filters = ['R','G','B']

    mask = np.zeros_like(IMG[:,:,0])
    for ci, filt in enumerate(filters):
        print("Processing %s filter"%filt)
        fnames = sorted(glob.glob('%s/*_proj_%s_*.nc'%(folder, filt)))

        lat = np.zeros((128, 1648, len(fnames)))
        lon = np.zeros((128, 1648, len(fnames)))
        img = np.zeros((128, 1648, len(fnames)))

        ''' load the map projected data '''
        for i, fi in enumerate(fnames):
            data = nc.Dataset(fi, 'r')

            lat[:,:,i] = data.variables['lat'][:]
            lon[:,:,i] = data.variables['lon'][:]
            img[:,:,i]  = data.variables['img'][:]
            
            flatlat = lat[:,:,i].flatten()
            flatlon = lon[:,:,i].flatten()

            maski   = flatlat != -1000.
            flatlat = flatlat[maski]
            flatlon = flatlon[maski]
            
        lat = lat.flatten()
        lon = lon.flatten()
        img = img.flatten()

        ''' remove pixels that were not projected '''
        invmask = np.where((lat == -1000.))[0]
        lat = np.delete(lat, invmask)
        lon = np.delete(lon, invmask)
        img = np.delete(img, invmask)

        ''' 
            figure out pixels in the projected image that
            were not in the original image
        '''

        hist,_,_ = np.histogram2d(lon, lat, bins=(newlonbins, newlatbins))

        hist = hist.T

        maskrow, maskcol = np.where(hist > 0)
        
        print("Cleaning up...")
        ''' 
            mask out artificafts where the value is extrapolated
            from pixels that are too far away
        '''
        for jj in range(maskrow.shape[0]):
            progress = jj/maskrow.shape[0]
            print("\r[%-20s] %d/%d"%(int(progress*20)*'=', jj, maskrow.shape[0]), end='')
            rowi = maskrow[jj]
            coli = maskcol[jj]
            mask[(rowi-padding):(rowi+padding),(coli-padding):(coli+padding)] = 1.
        print()       
        IMG[:,:,ci] = griddata((lon, lat), img, (LON, LAT), method='cubic').T
        #interpfunc = interp2d(lon, lat, img, kind='linear', bounds_error=True, fill_value=0.)
        #IMG[:,:,ci] = interpfunc(newlon, newlat)
    for i in range(3):
        IMG[:,:,i] = IMG[:,:,i]*mask[:]

    ''' save the mask and the raw pixel values '''
    plt.imsave('mask.png', mask, cmap='gray', origin='lower')
    np.save("%s/raw.npy"%folder, IMG, allow_pickle=False)

    IMG[np.isnan(IMG)] = 0.
    IMG[IMG<0.] = 0.

    ''' normalize the image by the 95% percentile '''
    IMG = IMG/(np.percentile(IMG[IMG>0.], 95))

    plt.imsave('mosaic_RGB.png', IMG, origin='lower')
    
    return

