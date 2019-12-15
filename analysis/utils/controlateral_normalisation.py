import numpy as np
from scipy.ndimage import gaussian_filter


def normalise_by_contralateral_median(data):
    '''
    Normalise an image by dividing every voxel by the mean value of the contralateral side
    :param
    :return:
    '''
    normalised_data = np.zeros(data.shape, dtype=np.float64)
    x_center = data.shape[0] // 2

    # normalise left side
    right_side = data[x_center:]
    clipped_right_side = np.clip(right_side, np.percentile(right_side, 1), np.percentile(right_side, 99))
    right_side_median = np.nanmean(clipped_right_side)
    normalised_data[:x_center] = np.divide(data[:x_center], right_side_median)

    # normalise right side
    left_side = data[:x_center]
    clipped_left_side = np.clip(left_side, np.percentile(left_side, 1), np.percentile(left_side, 99))
    left_side_median = np.nanmean(clipped_left_side)
    normalised_data[x_center:] = np.divide(data[x_center:], left_side_median)

    return normalised_data

def normalise_by_contralateral_voxel(data):
    '''
    Normalise an image by dividing every voxel by the voxel value of the contralateral side
    :param
    :return:
    '''
    normalised_data = data.copy()
    x_center = data.shape[0] // 2
    left_side_set_off = x_center
    if data.shape[0] % 2 == 0:
        # if number voxels along x is even, split in the middle
        right_side_set_off = x_center
    else:
        # if number voxels along x is uneven leave out the middle voxel line
        right_side_set_off = x_center + 1


    # normalise left side
    right_side = data[right_side_set_off:]
    flipped_right_side = right_side[::-1, ...]
    normalised_data[:left_side_set_off] = np.divide(data[:left_side_set_off], flipped_right_side)

    # normalise right side
    left_side = data[:left_side_set_off]
    flipped_left_side = left_side[::-1, ...]
    normalised_data[right_side_set_off:] = np.divide(data[right_side_set_off:], flipped_left_side)

    if data.shape[0] % 2 != 0:
        x_para_median_slices_mean = np.expand_dims(np.nanmean([data[x_center - 1], data[x_center + 1]], axis=0), axis=0)
        normalised_data[x_center] = np.divide(data[x_center], x_para_median_slices_mean)


    return normalised_data

def normalise_by_contralateral_region(data):
    '''
    Normalise an image by dividing every voxel by the voxel value of the contralateral side
    :param
    :return:
    '''
    normalised_data = data.copy()
    x_center = data.shape[0] // 2
    left_side_set_off = x_center
    if data.shape[0] % 2 == 0:
        # if number voxels along x is even, split in the middle
        right_side_set_off = x_center
    else:
        # if number voxels along x is uneven leave out the middle voxel line
        right_side_set_off = x_center + 1

    # normalise left side
    right_side = data[right_side_set_off:]
    flipped_right_side = right_side[::-1, ...]
    flipped_right_side_regions = gaussian_filter(flipped_right_side, sigma=7)
    normalised_data[:left_side_set_off] = np.divide(data[:left_side_set_off], flipped_right_side_regions)

    # normalise right side
    left_side = data[:left_side_set_off]
    flipped_left_side = left_side[::-1, ...]
    flipped_left_side_regions = gaussian_filter(flipped_left_side, sigma=7)
    normalised_data[right_side_set_off:] = np.divide(data[right_side_set_off:], flipped_left_side_regions)

    if data.shape[0] % 2 != 0:
        x_para_median_slices_mean = np.expand_dims(np.nanmean([data[x_center - 1], data[x_center + 1]], axis=0), axis=0)
        normalised_data[x_center] = np.divide(data[x_center], x_para_median_slices_mean)


    return normalised_data
