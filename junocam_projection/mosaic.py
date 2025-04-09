import numpy as np
from scipy.ndimage import gaussian_filter


def mosaic_median(maps: np.ndarray) -> np.ndarray:
    """
    Mosaic a set of maps by median-combining the data. Applies a percentile filter to remove regions with no data

    :param maps: an array of cylindrically projected maps (shape: [nfiles, height, width, 3])

    :return: mosaic
    """
    maps_clipped = maps.copy()
    maps_clipped[np.abs(maps) < np.percentile(maps[maps > 0], 1)] = np.nan
    mosaic = np.nanmedian(maps_clipped, axis=0)
    mosaic[~np.isfinite(mosaic)] = 0.
    return mosaic


def lowpass(data: np.ndarray, sigma: float):
    """
    Simple Gaussian filter with a size of sigma pixels. Accounts for zonal and meridional boundaries

    :param data: input data to be filtered (must be 2D)
    :param sigma: filter size in pixels

    :return: low-pass filtered image (Gaussian blurred image)
    """
    return gaussian_filter(data, sigma=sigma, mode=['nearest', 'wrap'])


def highpass(mapi: np.ndarray, sigma_cut: float, sigma_filter: float):
    """
    Apply a high pass filter by subtracting a Gaussian blurred image.
    Truncates the edges by applying a low-pass filter on the image footprint and trimming at 95% level

    :param mapi: input map (must be RGB)
    :param sigma_cut: filter width for edge detection and truncation
    :param sigma_filter: filter width for high-pass filter

    :returns:   - high pass filtered image
                - truncated footprint of the filtered image
    """
    Ls = mapi.mean(-1)

    # the footprint is the regions in the map where there is data
    # the minimum threshold is the 1-percentile value
    footprint_orig = np.asarray(Ls > np.percentile(Ls[Ls > 0], 1), dtype=np.float32)

    # we do a gaussian blur on this footprint so that we can figure out the edges
    footprint = np.repeat(lowpass(footprint_orig, sigma_cut)[:, :, np.newaxis], 3, axis=-1)

    # mask the edges
    footprint[footprint < 0.95] = 0
    footprint[footprint >= 0.95] = 1

    # we also apply a gaussian blur on the image brightness
    # this blur is the low-frequency signal
    low_freq = np.repeat(lowpass(Ls, sigma_filter)[:, :, np.newaxis], 3, axis=-1)

    # we filter the map on the low-frequency to get only the high-frequency components
    # and cut off the edges
    filtered = (mapi - low_freq) * footprint
    # filtered = filtered / np.percentile(filtered[filtered > 0], 99)
    return filtered, footprint


def blend_maps(maps: np.ndarray, sigma_filter: float = 40, sigma_cut: float = 50) -> np.ndarray:
    '''
    Combine the images using the USGS/ISIS no-seam technique. This works by applying a low-pass filter on a mosaic,
    and then combining the low-pass mosaic with a mosaic of the high-pass filtered data, which tends to remove seams efficiently.

    :param maps: input array of maps (shape: [nfiles, height, width, 3])
    :param sigma_filter: filter width for the high-pass/low-pass filter
    :param sigma_cut: filter width for edge detection (to truncate high frequency signal at the image edges)
    '''
    # convert to float32 to speed up the calculation
    maps = maps.astype(np.float32)

    # filter the input maps by truncating small values
    # and adjusting the input values to match each other
    for i, mapi in enumerate(maps):
        mapi = mapi / np.median(mapi[mapi > 0])
        mapi[mapi < 1e-4] = 0.
        maps[i] = mapi

    # open the file and load the variables
    nfiles, nlat, nlon, _ = maps.shape

    mosaic_initial = mosaic_median(maps)

    # apply a low pass on the initial mosaic
    lowpass_mosaic = np.zeros_like(mosaic_initial)
    for j in range(3):
        lowpass_mosaic[:, :, j] = lowpass(mosaic_initial[:, :, j], sigma_filter)

    # then high-pass filter the maps
    highpass_data = np.zeros_like(maps, dtype=np.float32)
    footprint = np.zeros((nlat, nlon))
    for i, mapi in enumerate(maps):
        highpass_data[i], footprinti = highpass(mapi, sigma_cut, sigma_filter)
        footprint += footprinti.mean(-1)

        # trim the edges again since we tend to end up with certain peaks
        highpass_data[i][highpass_data[i] > np.percentile(highpass_data[i][np.abs(highpass_data[i]) > 0], 99)] = 0.

    footprint = np.clip(footprint, 0, 1)

    # create a mosaic using the high-pass data
    mosaic_highpass = mosaic_median(highpass_data)

    # the final mosaic is the combination of the low-pass mosaic and the high-pass mosaic
    return (mosaic_highpass + lowpass_mosaic) * np.repeat(footprint[:, :, np.newaxis], 3, axis=2)
