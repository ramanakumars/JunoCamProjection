import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm


def blend_maps(maps: np.ndarray, blur_sigma: float = 50, trim_threshold: float = 0.25) -> np.ndarray:
    '''
    Blend a series of projected maps (all on the same coordinate system) together to reduce seams

    :param maps: A numpy array containing each individual JunoCam image projected on the same coordinate system
    :param blur_sigma: the Gaussian blur radius in pixels for removing large-scale luminance gradient (should be larger than the largest scale feature that needs to be resolved)
    :param trim_threshold: the standard deviation threshold above which to trim pixels. This is useful for reducing color noise. A large threshold will account for pixels where one color dominates over the others, e.g., when one filter is saturated)

    :return: the blended image in the same size as the original set of images
    '''

    # open the file and load the variables
    nfiles, ny, nx, _ = maps.shape
    filtered = np.zeros_like(maps, dtype=np.float32)

    for i in tqdm.tqdm(range(nfiles), desc='Applying filter'):
        mapi = maps[i]
        Ls = mapi.mean(-1)

        # find the image footprint for each map
        footprint = np.asarray(Ls > np.percentile(Ls, 1), dtype=np.float32)

        # we will blur this so we can trim the edges after we apply the high-pass filter
        footprint = np.repeat(gaussian_filter(footprint, sigma=blur_sigma, mode=['nearest', 'wrap'])[:, :, np.newaxis], 3, axis=-1)
        footprint[footprint < 0.8] = 0
        footprint[footprint >= 0.8] = 1

        # get the low frequency luminance gradient by applying a Gaussian blur on the luminance
        low_freq = np.repeat(gaussian_filter(Ls, sigma=blur_sigma, mode=['nearest', 'wrap'])[:, :, np.newaxis], 3, axis=-1)

        # the filtered image is divided by the low frequency data (i.e., retains high frequency info)
        filtered[i] = mapi / (low_freq + 1e-6) * footprint

        # flatten the image histogram
        filtered[i] = filtered[i] / np.percentile(filtered[i][filtered[i] > 0], 99)

    filtered_Ls = filtered.mean(-1)

    IMG = np.zeros((ny, nx, 3))
    alpha = np.zeros_like(filtered[:, :, :, 0])
    cut_threshold = np.nanpercentile(filtered_Ls[filtered_Ls > 0.0].flatten(), 1)

    for kk in range(nfiles):
        # find the standard deviation in the image axes, and cut off pixels which are above the
        # trim threshold for color variance
        std = np.nanstd(filtered[kk], axis=(-1))
        mask_k = (filtered_Ls[kk] > cut_threshold) & (std / filtered_Ls[kk] < trim_threshold)
        # weight each image by its relative brightness wrt to the global mean
        if len(filtered_Ls[kk, :, :][mask_k]) > 0:
            alpha[kk][mask_k] = np.nanmean(filtered_Ls[kk, :, :][mask_k])
            alpha[kk][alpha[kk, :] != 0] = 1. / alpha[kk, :][alpha[kk, :] != 0.]

    alpha_sum = np.sum(alpha, axis=0)
    alpha_sum[alpha_sum == 0] = np.nan

    IMGs_sub = filtered.copy()
    for c in range(3):
        IMGs_sub[:, :, :, c] = IMGs_sub[:, :, :, c] * alpha

    # remove really dim pixels which can occur on the edges and will pull the median down
    IMGs_sub[IMGs_sub <= np.percentile(filtered_Ls, 40)] = np.nan

    IMG = np.nanmedian(IMGs_sub, axis=0)

    IMG[~np.isfinite(IMG)] = 0.

    return IMG
