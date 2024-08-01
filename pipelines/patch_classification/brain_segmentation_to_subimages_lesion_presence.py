from analysis_tools.utils.masked_rolling_subimage_transformer import MaskedRollingSubImageTransformer
from tqdm import tqdm
import numpy as np


def brain_segmentation_to_subimages_lesion_presence(X, y, mask, width, stride=None, batch_size=10):
    """
    Convert a brain dataset to masked subimages and attribute lesion presence to each subimage.
    :param X: brain dataset (n_i, n_x, n_y, n_z)
    :param y: brain segmentation (n_i, n_x, n_y, n_z)
    :param mask: brain mask (n_i, n_x, n_y, n_z)
    :param width: int or list of ints detailing width of subimages
    :param stride: int or list of ints detailing stride for subimage sampling. Optional, defaults to width.
    :param batch_size: int. Optional, defaults to 10.
    :return: subimages: list(ndarray) (n_i) (n_subimages, width, width, width),
             lesion_presence: list(ndarray) (n_i) (n_subimages)
    """
    if isinstance(width, int):
        width = [width, width, width]
    if stride is None:
        stride = width
    if isinstance(stride, int):
        stride = [stride, stride, stride]


    masked_subimages_lesion_presence = []
    masked_subimages = []
    for batch_offset in tqdm(range(0, X.shape[0], batch_size)):
        X_batch = X[batch_offset:batch_offset + batch_size]
        y_batch = y[batch_offset:batch_offset + batch_size]
        mask_batch = mask[batch_offset:batch_offset + batch_size]
        masked_subimage_transformer = MaskedRollingSubImageTransformer(mask=mask_batch, width_list=[width],
                                                                       padding='valid', stride=stride)
        batch_masked_subimages = masked_subimage_transformer.fit_transform(X_batch)
        batch_masked_subimages_lesion_segmentation = masked_subimage_transformer.fit_transform(y_batch)
        width_idx = 0  # only one width is used for dataset creation (but transformer returns list of widths)
        batch_masked_subimages_lesion_presence = [
            np.any(batch_masked_subimages_lesion_segmentation[subj_idx][width_idx], axis=(-3, -2, -1))
            for subj_idx in range(X_batch.shape[0])]
        masked_subimages.append(batch_masked_subimages)
        masked_subimages_lesion_presence.append(batch_masked_subimages_lesion_presence)

    return np.squeeze(np.concatenate(masked_subimages)), np.concatenate(masked_subimages_lesion_presence)
