import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array


class ImageToPointCloud(BaseEstimator, TransformerMixin):
    """Represent active pixels in 2D/3D binary images as points in 2D/3D space.
    The coordinates of each point is calculated as follow. For each activated
    pixel, assign coordinates that are the pixel position on this image. All
    deactivated pixels are given infinite coordinates in that space.
    This transformer is meant to transform a collection of images to a point
    cloud so that collection of point clouds-based persistent homology module
    can be applied.
    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    Attributes
    ----------
    mesh_ : ndarray, shape (n_pixels_x * n_pixels_y [* n_pixels_z], \
        n_dimensions)
        Mesh image for which each pixel value is its coordinates in a
        `n_dimensions` space, where `n_dimensions` is the dimension of the
        images of the input collection. Set in meth:`fit`.
    See also
    --------
    gtda.homology.VietorisRipsPersistence, gtda.homology.SparseRipsPersistence,
    gtda.homology.EuclideanCechPersistence
    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _embed(self, X):
        Xpts = np.stack([self.mesh_ for _ in range(X.shape[0])]) * 1.0
        Xpts[np.logical_not(X.reshape((X.shape[0], -1))), :] += np.inf
        # Xpts = np.asarray([np.asarray(np.where(x)).T for x in X])
        return Xpts

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
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        n_dimensions = len(X.shape) - 1
        axis_order = [2, 1, 3]
        mesh_range_list = [np.arange(0, X.shape[i])
                           for i in axis_order[:n_dimensions]]

        self.mesh_ = np.flip(np.stack(np.meshgrid(*mesh_range_list),
                                      axis=n_dimensions),
                             axis=0).reshape((-1, n_dimensions))

        return self

    def transform(self, X, y=None):
        """For each collection of binary images, calculate the corresponding
        collection of point clouds based on the coordinates of activated
        pixels.
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
        Xt : ndarray, shape (n_samples, n_pixels_x * n_pixels_y [* n_pixels_z],
            n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in a `n_dimensions` dimensional space.
        """
        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._embed)(X[s])
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)
        return Xt
