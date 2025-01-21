# Authors: The scikit-learn developers and Nikita Lazarev
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warnings
from sklearn.utils import _array_api
from sklearn.utils.validation import validate_data
from sklearn.utils._array_api import get_namespace, get_namespace_and_device
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_random_state
import numpy as np
from scipy import sparse, stats
from sklearn.preprocessing._data import BOUNDS_THRESHOLD

from fqt.interpolate import interpolate_torch
from fqt.utils import resample

from sklearn import set_config

from typing import Literal, Optional


class FastQuantileTransformer(QuantileTransformer):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    For example visualizations, refer to :ref:`Compare QuantileTransformer with
    other scalers <plot_all_scaling_quantile_transformer_section>`.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int or None, default=10_000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.
        Disable subsampling by setting `subsample=None`.

        .. versionadded:: 1.5
           The option `None` to disable subsampling was added.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray of shape (n_quantiles, )
        Quantiles of references.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X)
    array([...])
    """

    def __init__(
            self,
            noise_distribution: Optional[Literal['uniform', 'normal', 'deterministic']] = None,
            replace_subsamples: bool = True,
            epsilon = 1e-5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.noise_distribution = noise_distribution
        self.replace_subsamples = replace_subsamples
        self.epsilon = epsilon

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        xp, _, device_ = get_namespace_and_device(X)

        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        random_state = check_random_state(random_state)

        if self.noise_distribution == 'normal':
            X += xp.asarray(random_state.normal(0.0, self.epsilon, X.shape), dtype=X.dtype, device=device_)
        elif self.noise_distribution == 'uniform':
            X += xp.asarray(random_state.uniform(-self.epsilon, self.epsilon, X.shape), dtype=X.dtype, device=device_)

        n_samples, n_features = X.shape
        if self.subsample is not None and self.subsample < n_samples:
            # Take a subsample of `X`
            X = resample(
                X, n_samples=self.subsample, random_state=random_state, replace=self.replace_subsamples
            )

        self.quantiles_ = xp.nanquantile(X, self.references_, axis=0)

    # @sklearn.base._fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted tr ansformer.
        """
        if self.subsample is not None and self.n_quantiles > self.subsample:
            raise ValueError(
                "The number of quantiles cannot be greater than"
                " the number of samples used. Got {} quantiles"
                " and {} samples.".format(self.n_quantiles, self.subsample)
            )

        X = self._check_inputs(X, in_fit=True, copy=False)
        xp, _, device_ = get_namespace_and_device(X)
        n_samples = X.shape[0]

        if self.n_quantiles > n_samples:
            warnings.warn(
                "n_quantiles (%s) is greater than the total number "
                "of samples (%s). n_quantiles is set to "
                "n_samples." % (self.n_quantiles, n_samples)
            )
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = xp.linspace(0, 1, self.n_quantiles_, endpoint=True, dtype=X.dtype, device=device_)
        if sparse.issparse(X):
            if self.noise_distribution is not None:
                warnings.warn(
                    "'noise_distribution' takes no effect with"
                    " sparse matrix. It's ignored due to sparce property."
                    " Consider making preprocess by yourself or using dense data type."
                )
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        return self
    
    def _interp(self, x, Xp, fp):
        xp, _, device_ = get_namespace_and_device(x)

        if xp.__name__ in {"torch", "array_api_compat.torch"}:
            if str(device_) == 'cpu':
                return torch.tensor(np.interp(x, Xp, fp), dtype=x.dtype, device=device_)
            else:
                return interpolate_torch(x, Xp, fp)
        else:
            return np.interp(x, Xp, fp)

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""
        xp, _ = get_namespace(X_col)

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    if xp.__name__ in {'torch', 'array_api_compat.torch'}:
                        X_col = torch.distributions.normal.Normal(0, 1).cdf(X_col)
                    else:
                        X_col = np.array(stats.norm.cdf(X_col))
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~xp.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            if self.noise_distribution in {None, 'deterministic'}:
                # Interpolate in one direction and in the other and take the
                # mean. This is in case of repeated values in the features
                # and hence repeated quantiles
                #
                # If we don't do this, `only one extreme` of the duplicated is
                # used (the upper when we do ascending, and the
                # lower for descending). We take the mean of these two
                X_col[isfinite_mask] = 0.5 * (
                        self._interp(X_col_finite, quantiles, self.references_)
                        - self._interp(-X_col_finite, -xp.flipud(quantiles), -xp.flipud(self.references_))
                )
            else:
                X_col[isfinite_mask] = self._interp(X_col_finite, quantiles, self.references_)
        else:
            X_col[isfinite_mask] = self._interp(X_col_finite, self.references_, quantiles)

        if self.noise_distribution != 'deterministic': 
            X_col[upper_bounds_idx] = upper_bound_y
            X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    if xp.__name__ in {'torch', 'array_api_compat.torch'}:
                        X_col = torch.distributions.normal.Normal(0, 1).icdf(X_col)
                    else:
                        X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = xp.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        xp, _, device_ = get_namespace_and_device(X)
        if str(device_) != 'cpu' and device_ != None:
            X = X.cpu()
        X = validate_data(
            self,
            X,
            reset=in_fit,
            accept_sparse="csc",
            copy=copy,
            dtype=_array_api.supported_float_dtypes(xp),
            # only set force_writeable for the validation at transform time because
            # it's the only place where QuantileTransformer performs inplace operations.
            force_writeable=True if not in_fit else None,
            ensure_all_finite="allow-nan",
        )
        if xp.__name__ in {"torch", "array_api_compat.torch"}:
            X = X.to(device_)
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings TODO: fix
            if (
                    not accept_sparse_negative
                    and not self.ignore_implicit_zeros
                    and (sparse.issparse(X) and xp.any(X.data < 0))
            ):
                raise ValueError(
                    "QuantileTransformer only accepts non-negative sparse matrices."
                )

        return X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.array_api_support = True
        return tags

def fast_quantile_transform(X,
    *,
    axis=0,
    n_quantiles=1000,
    output_distribution="uniform",
    ignore_implicit_zeros=False,
    subsample=int(1e5),
    random_state=None,
    copy=True,
    noise_distrubition=None,
    replace_subsample=True,
    epsilon=1e-5,
):
    n = FastQuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=subsample,
        ignore_implicit_zeros=ignore_implicit_zeros,
        random_state=random_state,
        copy=copy,
        noise_distribution=noise_distrubition,
        replace_subsamples=replace_subsample,
        epsilon=epsilon
    )
    if axis == 0:
        X = n.fit_transform(X)
    else:  # axis == 1
        X = n.fit_transform(X.T).T
    return X