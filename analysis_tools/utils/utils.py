import numpy as np


def get_undersample_selector_array(y, mask=None):
    """
    Return boolean array with true for indices fitting a undersampled balance
    Useful for multidimensional arrays

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)
        mask : mask representing the areas where negative samples of y can be taken from

    Returns:
        selector : boolean with true indices retained after random undersampling
    """
    flat_labels = np.squeeze(y.reshape(-1, 1))
    flat_mask = None
    if mask is not None:
        print('Using a mask for sampling.')
        flat_mask = np.squeeze(mask.reshape(-1, 1))
    undersampled_indices, unselected_indices = index_undersample_balance(flat_labels, flat_mask)
    selector = np.full(flat_labels.shape, False)
    selector[undersampled_indices] = True
    selector = selector.reshape(y.shape)

    return selector


def index_undersample_balance(y, mask=None):
    """
    Find indices fitting a undersampled balance

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)
        mask : mask representing the areas where negative samples of y can be taken from

    Returns:
        undersampled_indices : indices retained after random undersampling
        unselected_indices : ind9ces rejected after random undersampling
    """
    print('Undersampling Ratio 1:1')
    n_pos = np.sum(y)
    if mask is not None:
        # Only take negatives out of the mask (positives are always in the mask)
        masked_negatives = np.all([y == 0, mask == 1], axis=0)
        neg_indices = np.squeeze(np.argwhere(masked_negatives))
    else:
        neg_indices = np.squeeze(np.argwhere(y == 0))
    pos_indices = np.squeeze(np.argwhere(y == 1))
    randomly_downsampled_neg_indices = np.random.choice(neg_indices, int(n_pos), replace=False)
    undersampled_indices = np.concatenate([pos_indices, randomly_downsampled_neg_indices])
    unselected_indices = np.setdiff1d(neg_indices, randomly_downsampled_neg_indices)

    return undersampled_indices, unselected_indices

def pad_to_even_dimension_x(data):
    """

    :param data: brain volumes with shape (n_samples, n_x, n_y, n_z, [n_c]). Data will be padded on the x-axis.
    :return: np.ndarray padded data
    """
    provisory_x_center = data.shape[1] // 2
    provisory_right_side = data[:, provisory_x_center:]
    provisory_left_side = data[:, :provisory_x_center]
    added_channel_dim = False
    if data.ndim == 4:
        data = np.expand_dims(data, axis=-1)
        added_channel_dim = True
    assert data.ndim == 5, print('Brain volumes should be of shape (n_samples, n_x, n_y, n_z, [n_c])')
    if np.count_nonzero(provisory_right_side > 0) > np.count_nonzero(provisory_left_side > 0):  # ie most brain tissue is on right side of image
        print('Padding right.')
        padded_data = np.pad(data, ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), constant_values=0)  # pad right side of image with 0
    else:
        print('Padding left.')
        padded_data = np.pad(data, ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)), constant_values=0)  # pad left side of image with 0
    if added_channel_dim:
        padded_data = padded_data[..., 0]
    return padded_data

def brain_to_hemispheres(data: np.ndarray, uniform_side: bool = True):
    """
    Split the provided brain volumes into two equally sized hemispheres along a mid-saggital plane.
    :param data: brain volumes with shape (n_samples, n_x, n_y, n_z, [n_c]). Data will be split along the x-axis.
    :param uniform_side: if this option is selected, all hemispheres will be oriented in the same way (ie. left hemisphere [right hemisphere in radiological denomination]).
    :return: np.ndarray hemispheres
    """
    x_center = data.shape[1] // 2

    # split brain (here in image denomination, anatomical denomination would be the contrary)
    right_side = data[:, x_center:]
    left_side = data[:, :x_center]

    # if hemispheres of uneven dimension, shave on in the middle to avoid midline information cross-over
    if right_side.shape[1] > left_side.shape[1]:
        right_side = right_side[:, 1:]
    elif right_side.shape[1] < left_side.shape[1]:
        left_side = left_side[:, :-1]

    if uniform_side:
        transposed_right_side = np.flip(right_side, axis=1)
        hemispheres = np.concatenate((left_side, transposed_right_side), axis=0)
        return hemispheres
    else:
        hemispheres = np.concatenate((left_side, right_side), axis=0)
        return hemispheres
