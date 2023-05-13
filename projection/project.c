#include "cspice/include/SpiceUsr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define JUPITER 599
#define FRAME_HEIGHT 128
#define FRAME_WIDTH 1648
#define PI M_PI

double CY[3] = {158.48, 3.48, -151.52};

double aperture = 1.;

/* from https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/ik/juno_junocam_v03.ti
 */
double pixel_size = 7.4e-6;
double focal_length = 0.001095637;

struct Camera {
  double k1, k2, cx, cy, flength, psize, f1, time_bias, iframe_delay;
};

// subtract two 3D vectors: out = x - y
void subtract3D(double *x, double *y, double *out) {
  for (int i = 0; i < 3; i++) {
    out[i] = x[i] - y[i];
  }
}

// 3D matrix dot product: out = A dot b
void matmul3D(double *A, double *b, double *out) {
  for (int i = 0; i < 3; i++) {
    out[i] = 0.;
    for (int j = 0; j < 3; j++) {
      out[i] += A[i * 3 + j] * b[j];
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
  camera->psize = 0.0074;
  camera->f1 = camera->flength / camera->psize;

  camera->time_bias = 0.06188;
  camera->iframe_delay = 0.001;

  return 1;
}

void distort(struct Camera *camera, double *c, double *campos) {
  double xd, yd, r2, dr;
  xd = c[0];
  yd = c[1];
  r2 = xd * xd + yd * yd;
  dr = 1 + camera->k1 * r2 + camera->k2 * r2 * r2;
  xd *= dr;
  yd *= dr;

  campos[0] = xd;
  campos[1] = yd;
}

void vec2pix(struct Camera *camera, double *v, double *pix) {
  double alpha, x, y;
  double cam[2];
  double campos[2];

  alpha = v[2] / camera->f1;
  cam[0] = v[0] / alpha;
  cam[1] = v[1] / alpha;
  distort(camera, cam, campos);
  x = campos[0] + camera->cx;
  y = campos[1] + camera->cy;

  pix[0] = x;
  pix[1] = y;
}

void undistort(struct Camera *camera, double *cam, double *ucam) {
  register double xd, yd, r2, dr;
  xd = cam[0];
  yd = cam[1];
  for (int i = 0; i < 5; i++) {
    r2 = xd * xd + yd * yd;
    dr = 1. + camera->k1 * r2 + camera->k2 * r2 * r2;
    xd = cam[0] / dr;
    yd = cam[1] / dr;
  }
  ucam[0] = xd;
  ucam[1] = yd;
}

void pix2vec(struct Camera *camera, double *px, double *vec) {
  double cam[2], ucam[2];

  cam[0] = px[0] - camera->cx;
  cam[1] = px[1] - camera->cy;
  undistort(camera, cam, ucam);
  vec[0] = ucam[0];
  vec[1] = ucam[1];
  vec[2] = camera->f1;
}

int *get_image_mask(double *lat, double *lon, int nlat, int nlon, double *et,
                    int nframes) {
  struct Camera camera[3];

  initialize_camera(&camera[0], 0);
  initialize_camera(&camera[1], 1);
  initialize_camera(&camera[2], 2);

  int *mask;
  double surface_point[3], *vec, junocam_vec[3], pix[2], jup2cam_mat[3][3],
      *jup2cam, *scloc, state[6], lt;
  int count;

  jup2cam = malloc((nframes * 9) * sizeof(double));
  scloc = malloc((nframes * 3) * sizeof(double));
  vec = malloc((nframes * 3) * sizeof(double));

  for (int i = 0; i < nframes; i++) {
    // calculate the spacecraft position
    spkezr_c("JUNO", et[i], "IAU_JUPITER", "CN+S", "JUPITER", state, &lt);

    // copy over the position data
    scloc[i * 3 + 0] = state[0];
    scloc[i * 3 + 1] = state[1];
    scloc[i * 3 + 2] = state[2];

    // find the transform frame
    pxform_c("IAU_JUPITER", "JUNO_JUNOCAM", et[i], jup2cam_mat);

    // flatten the matrix to be used later
    for (int jj = 0; jj < 3; jj++) {
      for (int ii = 0; ii < 3; ii++) {
        jup2cam[i * 9 + jj * 3 + ii] = jup2cam_mat[jj][ii];
      }
    }
  }

  mask = malloc((nlat * nlon) * sizeof(int));

  for (int jj = 0; jj < nlat; jj++) {
    double latj, loni;
    if ((jj % 100) == 0)
      fprintf(stdout, "\r %4d/%4d ", jj, nlat);
    latj = lat[jj];
    for (int ii = 0; ii < nlon; ii++) {
      mask[jj * nlon + ii] = 0;
      loni = lon[ii];

      // find the vector to the point on the surface
      // in the Jupiter frame
      srfrec_c(JUPITER, loni, latj, surface_point);

      // get the vectors for all frames now
      for (int i = 0; i < nframes; i++) {
        // the vector from the spacecraft to the surface
        // in the JUPITER frame
        subtract3D(surface_point, &scloc[i * 3], junocam_vec);

        // project this vector into the JUNOCAM frame
        matmul3D(&jup2cam[i * 9], junocam_vec, &vec[i * 3]);
      }

      count = 0;
      // loop over each camera
      for (int n = 0; n < 3; n++) {
        // check if any frame saw this this point in the given color
        for (int i = 0; i < nframes; i++) {
          // find the pixel coordinate corresponding
          // to this vector
          vec2pix(&camera[n], &vec[i * 3], pix);

          // check if the pixel falls inside the photoactive area
          if ((pix[0] >= 23.5) & (pix[0] < 1630.5) & (pix[1] >= 0.5) &
              (pix[1] < 127.5)) {
            count++;
            break;
          }
        }
      }
      if (count == 3) {
        mask[jj * nlon + ii] = 1;
      }
    }
    fflush(stdout);
  }
  printf("\n");

  return mask;
}

void furnish(char *file) { furnsh_c(file); }

void process(double eti, int cam, double *cam2jup, double *lon, double *lat,
             double *inclin, double *emis, double *fluxcal) {
  double pix[2], spoint[3], srfvec[3], pixvec[3], pos_jup[3], scloc[3];
  double disti, loni, lati, phase, inc, emission, trgepoch, lt, plate_scale,
      ang_size, footprint;
  int found;

  plate_scale = (1. / focal_length) / (pixel_size);

  struct Camera camera;

  initialize_camera(&camera, cam);

  spkpos_c("JUNO", eti, "IAU_JUPITER", "CN+S", "JUPITER", scloc, &lt);

  for (int jj = 0; jj < FRAME_HEIGHT; jj++) {
    for (int ii = 23; ii < FRAME_WIDTH - 17; ii++) {
      // convert the pixel coordinate to a vector
      // in the JUNOCAM frame
      pix[0] = (double)ii;
      pix[1] = (double)jj;
      pix2vec(&camera, pix, pixvec);

      // transform this to the IAU_JUPITER coordinate
      matmul3D(cam2jup, pixvec, pos_jup);

      // find an intercept between the vector and the Jovian "surface"
      sincpt_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO",
               "IAU_JUPITER", pos_jup, spoint, &trgepoch, srfvec, &found);

      if (found) {
        // if the intercept exists, calculate the
        // coordinates of the intercept
        reclat_c(spoint, &disti, &loni, &lati);

        // calculate the illumination angle to do
        // solar flux correction
        ilumin_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO",
                 spoint, &trgepoch, srfvec, &phase, &inc, &emission);

        // lambertian correction
        // scorri = 1./inclin;
        // scorri = 2.*inclin/(emis + inclin);

        lati = lati * 180. / M_PI;
        loni = loni * 180. / M_PI;

        disti = (spoint[0] - scloc[0]) * (spoint[0] - scloc[0]);
        disti += (spoint[1] - scloc[1]) * (spoint[1] - scloc[1]);
        disti += (spoint[2] - scloc[2]) * (spoint[2] - scloc[2]);

        ang_size = M_PI * (aperture / disti) * (aperture / disti);
        footprint = (plate_scale * disti / cos(emission));

        fluxcal[jj * FRAME_WIDTH + ii] = 1. / (ang_size * footprint);

      } else {
        inc = 1000.;
        emission = 1000.;
        loni = -1000.;
        lati = -1000.;
      }

      // save to the output array
      inclin[jj * FRAME_WIDTH + ii] = inc;
      emis[jj * FRAME_WIDTH + ii] = emission;
      lat[jj * FRAME_WIDTH + ii] = lati;
      lon[jj * FRAME_WIDTH + ii] = loni;
    }
  }
}

void project_midplane(double eti, int cam, double tmid, double *lon,
                      double *lat, double *incid, double *emis, double *coords) {
  double pix[2], pix_transformed[2], spoint[3], srfvec[3], pixvec[3], scloc[3],
      vec_transformed[3], vec_iau[3], temp[3][3], *pxfrm_mid, *pxfrm_iau, disti,
      loni, lati, phase, inc, emission, trgepoch, lt, plate_scale, ang_size,
      footprint;
  int found;

  struct Camera camera, cam0;
  initialize_camera(&camera, cam);
  initialize_camera(&cam0, 1);

  pxfrm_mid = malloc(9 * sizeof(double));
  pxfrm_iau = malloc(9 * sizeof(double));

  spkpos_c("JUNO", eti, "IAU_JUPITER", "CN+S", "JUPITER", scloc, &lt);

  pxfrm2_c("JUNO_JUNOCAM", "JUNO_JUNOCAM", eti, tmid, temp);
  for (int jj = 0; jj < 3; jj++) {
    for (int ii = 0; ii < 3; ii++) {
      pxfrm_mid[jj * 3 + ii] = temp[jj][ii];
    }
  }

  pxform_c("JUNO_JUNOCAM", "IAU_JUPITER", eti, temp);
  for (int jj = 0; jj < 3; jj++) {
    for (int ii = 0; ii < 3; ii++) {
      pxfrm_iau[jj * 3 + ii] = temp[jj][ii];
    }
  }

  for (int jj = 0; jj < FRAME_HEIGHT; jj++) {
    for (int ii = 0; ii < FRAME_WIDTH; ii++) {
      pix[0] = ii;
      pix[1] = jj;
      pix2vec(&camera, pix, pixvec);

      matmul3D(pxfrm_mid, pixvec, vec_transformed);
      matmul3D(pxfrm_iau, pixvec, vec_iau);

      // find an intercept between the vector and the Jovian "surface"
      sincpt_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO",
               "IAU_JUPITER", vec_iau, spoint, &trgepoch, srfvec, &found);

      if (found) {
        // find the coordinate of the surface point
        reclat_c(spoint, &disti, &loni, &lati);

        // calculate the illumination angle to do
        // solar flux correction
        ilumin_c("Ellipsoid", "JUPITER", eti, "IAU_JUPITER", "CN+S", "JUNO",
                 spoint, &trgepoch, srfvec, &phase, &inc, &emission);

        // lambertian correction
        // scorri = 1./inclin;
        // scorri = 2.*inclin/(emis + inclin);

        disti = (spoint[0] - scloc[0]) * (spoint[0] - scloc[0]);
        disti += (spoint[1] - scloc[1]) * (spoint[1] - scloc[1]);
        disti += (spoint[2] - scloc[2]) * (spoint[2] - scloc[2]);

        vec2pix(&cam0, vec_transformed, pix_transformed);

        lat[jj * FRAME_WIDTH + ii] = lati * 180. / M_PI;
        lon[jj * FRAME_WIDTH + ii] = loni * 180. / M_PI;
        coords[(jj * FRAME_WIDTH + ii) * 2] = pix_transformed[0];
        coords[(jj * FRAME_WIDTH + ii) * 2 + 1] = pix_transformed[1];
        incid[jj * FRAME_WIDTH + ii] = inc;
        emis[jj * FRAME_WIDTH + ii] = emission;
      }
    }
  }
}
