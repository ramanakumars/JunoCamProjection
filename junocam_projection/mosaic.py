import numpy as np
import tqdm
from scipy.ndimage import uniform_filter
import numba


@numba.jit(nopython=True, cache=True)
def nanmedian_axis0(arr):
    """Faster implementation of np.nanpercentile

    This implementation always takes the percentile along axis 0.
    Uses numba to speed up the calculation by more than 7x.

    Function is equivalent to np.nanpercentile(arr, <percentiles>, axis=0)

    :param arr: Array to calculate percentiles for

    :return: Array with median data
    """
    shape = arr.shape
    arr = arr.reshape((arr.shape[0], -1))
    out = np.empty((arr.shape[1]))
    for i in range(arr.shape[1]):
        out[i] = np.nanmedian(arr[:, i])
    return out.reshape(shape[1:])


def mosaic_median(maps: np.ndarray) -> np.ndarray:
    """
    Mosaic a set of maps by median-combining the data. Applies a percentile filter to remove regions with no data

    :param maps: an array of cylindrically projected maps (shape: [nfiles, height, width, 3])

    :return: mosaic
    """
    maps_clipped = maps.copy()
    maps_clipped[np.abs(maps) < np.percentile(maps[maps > 0], 1)] = np.nan
    mosaic = nanmedian_axis0(maps_clipped)
    mosaic[~np.isfinite(mosaic)] = 0.
    return mosaic


def lowpass(data: np.ndarray, sigma: float):
    """
    Simple uniform filter with a size of sigma pixels. Accounts for zonal and meridional boundaries

    :param data: input data to be filtered (must be 2D)
    :param sigma: filter size in pixels

    :return: low-pass filtered image (blurred image)
    """
    return uniform_filter(data, size=sigma, mode=['nearest', 'wrap'])


def get_footprint(Ls: np.ndarray, sigma_cut: float):
    """
    Truncates the edges by applying a low-pass filter on the image footprint and trimming at 95% level

    :param Ls: input brightness channel of the image (essentially grayscale version of image)
    :param sigma_cut: filter width for edge detection (to truncate high frequency signal at the image edges)

    :return: footprint of the image with the edges truncated
    """
    # the footprint is the regions in the map where there is data
    # the minimum threshold is the 1-percentile value
    footprint_orig = np.asarray(Ls > np.percentile(Ls[Ls > 0], 1), dtype=np.float32)
    footprint = lowpass(footprint_orig, sigma_cut)

    # mask the edges
    footprint[footprint < 0.95] = 0
    footprint[footprint >= 0.95] = 1

    return footprint


def highpass(mapi: np.ndarray, sigma_filter: float):
    """
    Apply a high pass filter by subtracting a Gaussian blurred image.

    :param mapi: input map (must be RGB)
    :param sigma_filter: filter width for high-pass filter

    :return: high pass filtered image
    """
    Ls = mapi.mean(-1)

    # we also apply a gaussian blur on the image brightness
    # this blur is the low-frequency signal
    low_freq = np.repeat(lowpass(Ls, sigma_filter)[:, :, np.newaxis], 3, axis=-1)

    # we filter the map on the low-frequency to get only the high-frequency components
    # and cut off the edges
    filtered = mapi - low_freq

    return filtered


def highpass_luminance_correction(mapi: np.ndarray, sigma_filter: float = 50):
    """
    Flatten the illumination geometry by applying a high-pass filter on the luminance data

    :param mapi: input map (must be RGB)
    :param sigma_cut: filter width for edge detection (to truncate high frequency signal at the image edges)

    :return: high-pass filtered image
    """
    Ls = mapi.mean(-1)

    # we also apply a gaussian blur on the image brightness
    # this blur is the low-frequency signal
    low_freq = np.repeat(lowpass(Ls, sigma_filter)[:, :, np.newaxis], 3, axis=-1)

    # we filter the map on the low-frequency to get only the high-frequency components
    # and cut off the edges
    filtered = mapi / (low_freq + 1e-6)
    filtered[~np.isfinite(filtered)] = 0.

    return filtered


def blend_maps(maps: np.ndarray, sigma_luminance: float = 50, sigma_filter: float = 40, sigma_cut: float = 50) -> np.ndarray:
    '''
    Combine the images using the USGS/ISIS no-seam technique. This works by applying a low-pass filter on a mosaic,
    and then combining the low-pass mosaic with a mosaic of the high-pass filtered data, which tends to remove seams efficiently.

    :param maps: input array of maps (shape: [nfiles, height, width, 3])
    :param sigma_filter: filter width for the high-pass/low-pass filter
    :param sigma_cut: filter width for edge detection (to truncate high frequency signal at the image edges)
    '''
    # convert to float32 to speed up the calculation
    maps = maps.astype(np.float32)

    # open the file and load the variables
    nfiles, nlat, nlon, _ = maps.shape

    maps_filtered = np.zeros_like(maps)

    # filter the input maps by truncating small values
    # and adjusting the input values to match each other
    footprints = np.zeros((nfiles, nlat, nlon, 3))
    for i, mapi in enumerate(tqdm.tqdm(maps, desc="Applying luminance high pass")):
        footprints[i] = np.repeat(get_footprint(mapi.mean(-1), sigma_cut)[:, :, np.newaxis], 3, axis=2)

        mapi = highpass_luminance_correction(mapi, sigma_luminance)
        mapi = mapi / np.median(mapi[mapi > 0])
        mapi[mapi < 1e-4] = 0.
        mapi[~np.isfinite(mapi)] = 0.

        maps_filtered[i] = mapi

    footprint = np.clip(footprints.sum(0), 0, 1)

    print("Creating initial mosaic")
    mosaic_initial = mosaic_median(maps_filtered)

    # apply a low pass on the initial mosaic
    lowpass_mosaic = np.zeros_like(mosaic_initial)
    for j in range(3):
        lowpass_mosaic[:, :, j] = lowpass(mosaic_initial[:, :, j], sigma_filter)

    # then high-pass filter the maps
    highpass_data = np.zeros_like(maps, dtype=np.float32)
    for i, mapi in enumerate(tqdm.tqdm(maps_filtered, desc="Applying high-pass filter")):
        highpass_data[i] = highpass(mapi, sigma_filter) * footprints[i]

        # trim the edges again since we tend to end up with certain peaks
        if np.abs(highpass_data[i]).max() > 0:
            highpass_data[i][highpass_data[i] > np.percentile(highpass_data[i][np.abs(highpass_data[i]) > 0], 99)] = 0.

    # create a mosaic using the high-pass data
    print("Creating high-pass mosaic")
    mosaic_highpass = mosaic_median(highpass_data)

    # the final mosaic is the combination of the low-pass mosaic and the high-pass mosaic
    return (mosaic_highpass + lowpass_mosaic) * footprint
