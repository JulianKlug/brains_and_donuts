import numpy as np

def get_undersample_selector_array(y, mask = None):
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

def index_undersample_balance(y, mask = None):
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
        masked_negatives = np.all([y == 0, mask == 1], axis = 0)
        neg_indices = np.squeeze(np.argwhere(masked_negatives))
    else :
        neg_indices = np.squeeze(np.argwhere(y == 0))
    pos_indices = np.squeeze(np.argwhere(y == 1))
    randomly_downsampled_neg_indices = np.random.choice(neg_indices, int(n_pos), replace = False)
    undersampled_indices = np.concatenate([pos_indices, randomly_downsampled_neg_indices])
    unselected_indices = np.setdiff1d(neg_indices, randomly_downsampled_neg_indices)

    return (undersampled_indices, unselected_indices)