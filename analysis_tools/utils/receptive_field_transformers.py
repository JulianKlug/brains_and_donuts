import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .rolling_window import rolling_window

class ReceptiveFieldTransformer(BaseEstimator, TransformerMixin):
    """Obtain regularly spaced subimages belonging to the masked area of images in a collection.

    Parameters
    ----------
    width: width along every dimension [wx, wy, wz] - must be unpair
    """
    def __init__(self, width):
        self.width = width

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
        """
        return self

    def transform(self, X, y=None):
        """Extract the subimages of the given width

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z, n_c)
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            grayscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x * n_pixels_y [* n_pixels_z],
            n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in a `n_dimensions` dimensional space.
        """
        # pad all images to allow for an receptive field even at the borders

        no_channels = False
        if X.ndim < 5:
            X = np.expand_dims(X, axis=-1)
            no_channels = True
        assert X.ndim == 5, 'Input dimensions must be (n_samples, n_pixels_x, n_pixels_y, n_pixels_z  [, n_c])'

        padding_x, padding_y, padding_z = int((self.width[0] - 1) / 2), int((self.width[1] - 1) / 2), \
                                          int((self.width[2] - 1) / 2)
        padded_data = np.pad(X,
                             ((0, 0), (padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z), (0, 0)),
                             mode='constant', constant_values=0)

        Xt = rolling_window(padded_data, (0, self.width[0], self.width[1], self.width[2], 0))

        if no_channels:
            Xt = Xt[:, :, :, :, 0]

        return Xt


class MaskedReceptiveFieldTransformer(BaseEstimator, TransformerMixin):
    """Obtain regularly spaced subimages belonging to the masked area of images in a collection.

    Parameters
    ----------
    mask: mask defining the areas of the input images to use

    width_list: list of list of int or None, optional, default: ``None``
        List of different widths of the sub-images.
    """
    def __init__(self, mask, width_list=None):
        self.mask = mask
        self.width_list = width_list

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
        """
        return self

    def transform(self, X, y=None):
        """For each width listed in width_list, extract the subimages belonging to the mask

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            grayscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x * n_pixels_y [* n_pixels_z],
            n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in a `n_dimensions` dimensional space.
        """
        n_subjects = X.shape[0]

        ## Subimage creation
        # Subimages are created for every subject for every width setting in width list
        # Result: subject list of width list of voxel array: n_subj, n_widths, n_x, n_y, n_z, w_x, w_y, w_z
        X_subimages_width_list = [
            [ReceptiveFieldTransformer(width=width).fit_transform(X[np.newaxis, subj_index])
             for width in self.width_list] for subj_index in range(n_subjects)]

        ## Masking
        # Subimages are flattened to create a common voxel-level dimension and the subset defined by their mask is extracted
        # Result: subject list of width list of arrays : n_subjects, n_widths, n_voxels (differs for every subject), w_x, w_y, w_z
        X_masked_subimages = []
        for subj_index in range(n_subjects):
            subimages_per_width = []
            for width_index, width in enumerate(self.width_list):
                flat_subj_subimages = X_subimages_width_list[subj_index][width_index].reshape(-1, width[0], width[1],
                                                                                              width[2])
                flat_subj_mask = self.mask[subj_index].reshape(-1)
                subimages_per_width.append(flat_subj_subimages[flat_subj_mask])
            X_masked_subimages.append(subimages_per_width)
        Xt = X_masked_subimages

        return Xt
