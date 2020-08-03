import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pgtda.images import RollingSubImageTransformer

class MaskedRollingSubImageTransformer(BaseEstimator, TransformerMixin):
    """Obtain regularly spaced subimages belonging to the masked area of images in a collection.

    Parameters
    ----------
    mask: mask defining the areas of the input images to use

    width_list: list of list of int or None, optional, default: ``None``
        List of different widths of the sub-images. If ``None``, the effective width is taken as
        `3` across all dimensions.

    stride: list of int or None, default: ``None``
        Stride between sub-images. If ``None``, the effective stride is taken
        as `1` across all dimensions.

    padding: str or list of int, optional, default: ``'same'``
        Padding applied to the images in the input collection.
            - ``'same'``: Padding is calculated so that the output images have
                the same size as the input images.
            - ``'valid'``: No padding.
            - ``'full'``: Maximum padding so that there is at least one voxel
                of the orginal image with each subimage.

    activated : bool, optional, default: ``False``
        If ``True``, the padded pixels are activated. If ``False``, they are
        deactivated.

    periodic_dimensions : list of bool or None, optional, default: ``None``
        Periodicity of the boundaries along each of the axis, where
        ``n_dimensions`` is the dimension of the images of the collection. The
        boolean in the `d`th position expresses whether the boundaries along
        the `d`th axis are periodic. The default ``None`` is equivalent to
        passing ``numpy.zeros((n_dimensions,), dtype=np.bool)``, i.e. none of
        the boundaries are periodic.

    feature : bool, optional, default: ``False``
        If ``True``, the transformed array will be 2d of shape (n_samples, \
        n_features). If ``False``, the transformed array will preserve the
        shape of the transformer output array.

    Attributes
    ----------
    image_slices_ : list of slice
        List of slices corresponding to each subimage.

    mask_ = boolean ndarray of shape (n_samples, n_x, n_y, n_z)

    width_list_ : list of int ndarray of shape [n_samples, (width_x, width_y [, width_z]])
       Effective width along each of the axis. Set in :meth:`fit`.

    stride_ : int ndarray of shape (stride_x, stride_y [, stride_z])
       Effective stride along each of the axis. Set in :meth:`fit`.

    padding_ : int ndarray of shape (padding_x, padding_y [, padding_z])
       Effective padding along each of the axis. Set in :meth:`fit`.

    periodic_dimensions_ : boolean ndarray of shape (n_dimensions,)
       Effective periodicity of the boundaries along each of the axis.
       Set in :meth:`fit`.

    """
    def __init__(self, mask, width_list=None,
                 stride=None, padding='same', activated=False,
                 periodic_dimensions=None, feature=False):
        self.mask = mask
        self.width_list = width_list
        self.stride = stride
        self.padding = padding
        self.activated = activated
        self.periodic_dimensions = periodic_dimensions
        self.feature = feature

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
        """For each width listed in width_list, extract the subimages belonging

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
            [RollingSubImageTransformer(width=width, padding=self.padding, stride=self.stride, activated=self.activated,
                 periodic_dimensions=self.periodic_dimensions, feature=self.feature).fit_transform(X[np.newaxis, subj_index])
             for width in self.width_list] for subj_index in range(n_subjects)]

        ## Masking

        if self.padding == 'same':
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
        else:
            X_mask_subimages_width_list = [
            [RollingSubImageTransformer(width=width, padding=self.padding, stride=self.stride, activated=self.activated,
                 periodic_dimensions=self.periodic_dimensions, feature=self.feature).fit_transform(self.mask[np.newaxis, subj_index])
             for width in self.width_list] for subj_index in range(n_subjects)]
            X_masked_subimages = []
            for subj_index in range(n_subjects):
                subimages_per_width = []
                for width_index, width in enumerate(self.width_list):
                    flat_subj_subimages = X_subimages_width_list[subj_index][width_index].reshape(-1, width[0], width[1],
                                                                                                  width[2])
                    flat_subj_mask_subimages = X_mask_subimages_width_list[subj_index][width_index].reshape(-1, width[0] * width[1] * width[2])
                    flat_subj_mask_subimages_mask_presence = np.any(flat_subj_mask_subimages, axis=-1)
                    subimages_per_width.append(flat_subj_subimages[flat_subj_mask_subimages_mask_presence])
                X_masked_subimages.append(subimages_per_width)
            Xt = X_masked_subimages

        return Xt