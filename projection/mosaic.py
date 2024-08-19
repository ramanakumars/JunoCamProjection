import numpy as np
import healpy as hp
from skimage import color
import tqdm


class Mosaic:
    def __init__(self, fnames, n_side=512):
        self.fnames = fnames
        self.nframes = len(fnames)
        self.n_side = n_side

    def load_data(self):
        self.maps = np.zeros((len(self.fnames), hp.nside2npix(self.n_side), 3))

        for i, fname in enumerate(tqdm.tqdm(self.fnames)):
            map = np.load(fname)
            if hp.nside2npix(self.n_side) != map.shape[0]:
                map = hp.ud_grade(map.T, self.n_side).T

            map_lab = color.rgb2lab(map)

            # clean up the fringes. these are negative a* and b* in the CIELAB space
            map_lab[(map_lab[:, 1] < -0.001) | (map_lab[:, 2] < -0.001), 0] = 0

            self.maps[i, :] = color.lab2rgb(map_lab)
        self.maps[~np.isfinite(self.maps)] = 0.

    def stack(self, radius=2):
        m_normed = self.maps.copy()
        # first, norma
        for j in range(self.nframes):
            m_normed[j, :] = m_normed[j, :] / np.percentile(m_normed[j, :], 99)

        count = np.sum(np.min(m_normed, axis=-1) > 1.e-6, axis=0)
        m_hsv = color.rgb2hsv(m_normed)
        v_ave = np.mean(m_hsv[:, count > 0, 2])

        pix_inds = np.where(count > 0)[0]
        v_loc = np.zeros_like(m_normed[:, :, 0])
        vecs = np.asarray(hp.pix2vec(self.n_side, ipix=pix_inds)).T

        for i, vec in tqdm.tqdm(zip(pix_inds, vecs), total=len(pix_inds)):
            neighbours = hp.query_disc(self.n_side, vec=vec, radius=np.radians(radius))
            m_hsv_neigh = m_hsv[:, neighbours, 2]
            v_loc[:, i] = np.mean(m_hsv_neigh[:, count[neighbours] > 0], axis=1)

        m_new = np.zeros_like(m_normed[0, :])
        for i in range(3):
            m_new[:, i] = np.sum(m_normed[:, :, i] * (v_ave / v_loc), axis=0) / count
        m_new[~np.isfinite(m_new)] = 0

        return m_new
