#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include "cspice/include/SpiceUsr.h"

#define JUPITER 599
#define FRAME_HEIGHT 128
#define FRAME_WIDTH  1648
#define PI M_PI

double CY[3] = {158.48, 3.48, -151.52};

struct Camera {
	double k1, k2, cx, cy, flength, psize, f1, time_bias, iframe_delay;
};

// subtract two 3D vectors: out = x - y
void subtract3D(double *x, double *y, double *out) {
	for(int i=0; i<3; i++) {
		out[i] = x[i] - y[i];
	}
}

// 3D matrix dot product: out = A dot b
void matmul3D(double *A, double *b, double *out) {
	for(int i=0; i<3; i++) {
		out[i] = 0.;
		for(int j=0; j<3; j++) {
			out[i] += A[i*3+j]*b[j];
		}
	}
}

// set the camera constants and transform functions
// from https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/ik/juno_junocam_v03.ti
int initialize_camera(struct Camera *camera, int cam) {
	camera->k1 = -5.9624209455667325e-08;
	camera->k2 = 2.7381910042256151e-14;
	camera->cx = 814.21;
	camera->cy = CY[cam];

	camera->flength = 10.95637;
	camera->psize   = 0.0074;
	camera->f1  	= camera->flength/camera->psize;

	camera->time_bias 	= 0.06188;
	camera->iframe_delay = 0.001;

	return 1;
}

void distort(struct Camera *camera, double *c, double *campos) {
	double xd, yd, r2, dr;
	xd = c[0]; yd = c[1];
	r2 = xd*xd + yd*yd;
	dr = 1+camera->k1*r2 + camera->k2*r2*r2;
	xd *= dr;
	yd *= dr;

	campos[0] = xd;
	campos[1] = yd;
}

void vec2pix(struct Camera *camera, double *v, double *pix){
	double alpha, x, y;
	double cam[2];
	double campos[2];

	alpha = v[2]/camera->f1;
	cam[0] = v[0]/alpha;
	cam[1] = v[1]/alpha;
	distort(camera, cam, campos);
	x     = campos[0] + camera->cx;
	y     = campos[1] + camera->cy;

	pix[0] = x; pix[1] = y;
}

void undistort(struct Camera *camera, double *cam, double *ucam) {
	register double xd, yd, r2, dr;
	xd = cam[0];
	yd = cam[1];
	for(int i=0; i < 5; i++) {
		r2 = xd*xd + yd*yd;
		dr = 1. + camera->k1*r2 + camera->k2*r2*r2;
		xd = cam[0]/dr;
		yd = cam[1]/dr;
	}
	ucam[0] = xd;
	ucam[1] = yd;
}

void pix2vec(struct Camera *camera, double *px, double *vec){
	double cam[2], ucam[2];

	cam[0] = px[0] - camera->cx;
	cam[1] = px[1] - camera->cy;
	undistort(camera, cam, ucam);
	vec[0] = ucam[0];
	vec[1] = ucam[1];
	vec[2] = camera->f1;
}


int* get_image_mask(double *lat, double *lon, int nlat, int nlon, 
		double *et, int nframes) {
	struct Camera camera[3];
	
	initialize_camera(&camera[0], 0);
	initialize_camera(&camera[1], 1);
	initialize_camera(&camera[2], 2);

	int *mask;
	double surface_point[3], *vec, junocam_vec[3], pix[2], jup2cam_mat[3][3], 
		   *jup2cam, *scloc, state[6], lt;
	int count;

	jup2cam = malloc((nframes*9)*sizeof(double));
	scloc   = malloc((nframes*3)*sizeof(double));
	vec     = malloc((nframes*3)*sizeof(double));

	for(int i=0; i<nframes; i++) {
		// calculate the spacecraft position
		spkezr_c("JUNO", et[i], "IAU_JUPITER", "CN", "JUPITER", state, &lt);

		// copy over the position data
		scloc[i*3+0] = state[0];
		scloc[i*3+1] = state[1];
		scloc[i*3+2] = state[2];

		//find the transform frame
		pxform_c("IAU_JUPITER", "JUNO_JUNOCAM", et[i], jup2cam_mat);

		// flatten the matrix to be used later
		for(int jj=0; jj<3; jj++) {
			for(int ii=0; ii<3; ii++) {
				jup2cam[i*9+jj*3+ii] = jup2cam_mat[jj][ii];
			}
		}
	}

	mask = malloc((nlat*nlon)*sizeof(int));
	
	for(int jj=0; jj<nlat; jj++) {
		double latj, loni; 
		if((jj%100)==0) fprintf(stdout,"\r %4d/%4d ", jj, nlat);
		latj = lat[jj];
		for(int ii=0; ii<nlon; ii++) {
			mask[jj*nlon+ii] = 0;
			loni = lon[ii];

			// find the vector to the point on the surface
			// in the Jupiter frame
			srfrec_c(JUPITER, loni, latj, surface_point);
			

			// get the vectors for all frames now
			for(int i=0; i<nframes; i++) {
				// the vector from the spacecraft to the surface
				// in the JUPITER frame
				subtract3D(surface_point, &scloc[i*3], junocam_vec);

				// project this vector into the JUNOCAM frame
				matmul3D(&jup2cam[i*9], junocam_vec, &vec[i*3]);
			}
			
			count = 0;
			// loop over each camera
			for(int n=0; n<3; n++) {
				// check if any frame saw this this point in the given color
				for(int i=0; i<nframes; i++) {
					// find the pixel coordinate corresponding
					// to this vector
					vec2pix(&camera[n], &vec[i*3], pix);

					// check if the pixel falls inside the photoactive area
					if((pix[0]>=23.5)&(pix[0]<1630.5)&(pix[1]>=0.5)&(pix[1]<127.5)) {
						count++;
						break;
					}
				}
			}
			if(count == 3) {
				mask[jj*nlon+ii] = 1;
			}
		}
		fflush(stdout);
	}
	printf("\n");

	return mask;
}

void furnish(char *file) {
	furnsh_c(file);
}

void process_all(int height, int width, double start_time, double frame_delay, \
		double *lat, double *lon, double *solar_corr, double *scloc, double *et) {

	double disti, loni, lati, scorr, mu, mu0,
		   phase, inc, emission, trgepoch, lt, eti;
	double pix[2], pixvec[3], pos_jup[3], spoint[3], srfvec[3],
		jup2cam_mat[3][3], jup2cam[9];
	int nframes, found, start;

	// calculate the number of framelets in the image
	nframes = height/(FRAME_HEIGHT*3);

	// initialize the three cameras
	struct Camera cameras[3];

	initialize_camera(&cameras[0], 0);
	initialize_camera(&cameras[1], 1);
	initialize_camera(&cameras[2], 2);

	for(int i=0; i<nframes; i++) {
		// calculate the spacecraft ET and location for a given framelet
		eti = start_time + cameras[0].time_bias + \
			(frame_delay + cameras[0].iframe_delay)*( (double) i);

		spkpos_c("JUNO", eti, "IAU_JUPITER", "CN", "JUPITER", scloc+i*3, &lt);

		// calculate the transformation matrix for this frame
		pxform_c("JUNO_JUNOCAM", "IAU_JUPITER", eti, jup2cam_mat);

		// flatten the matrix to be used later
		for(int jj=0; jj<3; jj++) {
			for(int ii=0; ii<3; ii++) {
				jup2cam[jj*3+ii] = jup2cam_mat[ii][jj];
			}
		}

		// save the location and time to the output array
		//scloc[i*3+0] = state[0];
		//scloc[i*3+1] = state[1];
		//scloc[i*3+2] = state[2];

		et[i] = eti;

		for(int cam=0; cam<3; cam++) {
			printf("\rframe: %3d/%3d, camera: %d et: %.4f", i, nframes, cam, eti);

			// starting index for the current framelet
			start = FRAME_WIDTH*(3*FRAME_HEIGHT*i + cam*FRAME_HEIGHT);
	
			for(int jj=0; jj<FRAME_HEIGHT; jj++) {
				for(int ii=23; ii<FRAME_WIDTH-17; ii++) {
					// convert the pixel coordinate to a vector
					// in the JUNOCAM frame
					pix[0] = (double) ii;
					pix[1] = (double) jj;
					pix2vec(&cameras[cam], pix, pixvec);
					
					// transform this to the IAU_JUPITER coordinate
					matmul3D(jup2cam, pixvec, pos_jup);

					// find an intercept between the vector and the Jovian "surface"
					sincpt_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", "JUNO", \
							"IAU_JUPITER", pos_jup, spoint, &trgepoch, srfvec, &found);

					if(found) {			
						// if the intercept exists, calculate the 
						// coordinates of the intercept
				        reclat_c(spoint, &disti, &loni, &lati);

						// calculate the illumination angle to do 
						// solar flux correction
						ilumin_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", "JUNO", spoint, \
									&trgepoch, srfvec, &phase, &inc, &emission);

						mu  = cos(emission);
						mu0 = cos(inc);
						
						if(mu0 < PI/2.) {
							scorr = 2.*mu0/(mu + mu0);
						} else {
							scorr = 0.;
						}


						lati = lati*180./M_PI;
						loni = loni*180./M_PI;
					} else {
						scorr = 0.;
						loni  = -1000.;
						lati  = -1000.;
					}

					// save to the output array
					solar_corr[start + jj*FRAME_WIDTH + ii] = scorr;
					lat[       start + jj*FRAME_WIDTH + ii] = lati;
					lon[       start + jj*FRAME_WIDTH + ii] = loni;
					
				}
			}
			fflush(stdout);
		}
	}
	printf("\n");
	return;
}


void process(double eti, int cam, double *cam2jup, 
		double *lon, double *lat, double *inclin, double *emis) {
	double pix[2], spoint[3], srfvec[3], pixvec[3], pos_jup[3];
	double disti, loni, lati, phase, inc, emission, trgepoch;
	int found;


	struct Camera camera;

	initialize_camera(&camera, cam);

	for(int jj=0; jj<FRAME_HEIGHT; jj++) {
		for(int ii=23; ii<FRAME_WIDTH-17; ii++) {
			// convert the pixel coordinate to a vector
			// in the JUNOCAM frame
			pix[0] = (double) ii;
			pix[1] = (double) jj;
			pix2vec(&camera, pix, pixvec);
			
			// transform this to the IAU_JUPITER coordinate
			matmul3D(cam2jup, pixvec, pos_jup);

			// find an intercept between the vector and the Jovian "surface"
			sincpt_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", \
					"JUNO", "IAU_JUPITER", pos_jup,spoint, &trgepoch, srfvec, &found);

			if(found) {			
				// if the intercept exists, calculate the 
				// coordinates of the intercept
				reclat_c(spoint, &disti, &loni, &lati);

				// calculate the illumination angle to do 
				// solar flux correction
				ilumin_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN", "JUNO", spoint, \
							&trgepoch, srfvec, &phase, &inc, &emission);

				// lambertian correction
				// scorri = 1./inclin;
				// scorri = 2.*inclin/(emis + inclin);


				lati = lati*180./M_PI;
				loni = loni*180./M_PI;
			} else {
				inc      = 1000.;
				emission = 1000.;
				loni     = -1000.;
				lati     = -1000.;
			}

			// save to the output array
			inclin[jj*FRAME_WIDTH + ii] = inc;
			emis[  jj*FRAME_WIDTH + ii] = emission;
			lat[   jj*FRAME_WIDTH + ii] = lati;
			lon[   jj*FRAME_WIDTH + ii] = loni;
			
		}
	}

}
