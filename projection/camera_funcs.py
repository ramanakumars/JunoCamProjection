from .globals import *

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
        '''
            Convert from pixel coordinate to vector in the 
            JUNO_JUNOCAM reference frame

            Parameters
            ----------
            px : array-like
                x and y position of pixel centers in the camera

            Output
            ------
            v : numpy.ndarray
                vector in the JUNO_JUNOCAM reference frame
        '''
        camx = px[0] - self.cx
        camy = px[1] - self.cy
        cam = self.undistort([camx, camy])
        v   = np.asarray([cam[0], cam[1], self.f1])
        return v

    def undistort(self, c):
        '''
            Removes the barrel distortion in the JunoCam image

            Parameters
            ----------
            c : array-like
                x and y position of pixel centers in the camera

            Output
            ------
            xd : float
                x position of the pixel after removing barrel distortion
            yd : float
                y position of the pixel after removing barrel distortion
        '''
        xd, yd = c[0], c[1]
        for i in range(5):
            r2 = (xd**2. + yd**2.)
            dr = 1. + self.k1*r2 + self.k2*r2*r2
            xd = c[0]/dr
            yd = c[1]/dr
        return (xd, yd)

    def distort(self, c):
        '''
            Adds barrel distortion to the image

            Parameters
            ----------
            c : array-like
                x and y position of undistorted pixel centers in the camera

            Output
            ------
            xd : float
                x position of the pixel after adding barrel distortion
            yd : float
                y position of the pixel after adding barrel distortion
        '''
        xd, yd = c[0], c[1]
        r2 = (xd**2+yd**2)
        dr = 1+self.k1*r2+self.k2*r2*r2
        xd *= dr
        yd *= dr
        return [xd, yd]

    def vec2pix(self, v):
        '''
            Convert a vector in the JUNO_JUNOCAM reference frame
            to pixel coordinates on the plate

            Parameters
            ----------
            v : array-like
                vector in the JUNO_JUNOCAM reference frame

            Output
            ------
            x : float
                x-center of the pixel in the plate
            y : float
                y-center of the pixel in the plate
        '''
        alpha = v[2]/self.f1
        cam   = [v[0]/alpha, v[1]/alpha]
        cam   = self.distort(cam)
        x     = cam[0] + self.cx
        y     = cam[1] + self.cy
        return (x,y)