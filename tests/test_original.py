# Authors:
#
#          Giorgio Patrini
#
# License: BSD 3 clause

import numpy as np
import pytest
from scipy import sparse

from sklearn import datasets
from sklearn.preprocessing import (
    quantile_transform,
)
from sklearn.preprocessing._data import BOUNDS_THRESHOLD
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
)
from sklearn.utils.fixes import (
    CSC_CONTAINERS,
)
from .context import *

iris = datasets.load_iris()

# Make some data to be used many times
rng = np.random.RandomState(0)
n_features = 30
n_samples = 1000
offsets = rng.uniform(-1, 1, size=n_features)
scales = rng.uniform(1, 10, size=n_features)
X_2d = rng.randn(n_samples, n_features) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_features)
X_1col = X_2d[:, 0].reshape(n_samples, 1)
X_list_1row = X_1row.tolist()
X_list_1col = X_1col.tolist()


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_iris(csc_container):
    X = iris.data
    # uniform output distribution
    transformer = FastQuantileTransformer(n_quantiles=30)
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    # normal output distribution
    transformer = FastQuantileTransformer(n_quantiles=30, output_distribution="normal")
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    # make sure it is possible to take the inverse of a sparse matrix
    # which contain negative value; this is the case in the iris dataset
    X_sparse = csc_container(X)
    X_sparse_tran = transformer.fit_transform(X_sparse)
    X_sparse_tran_inv = transformer.inverse_transform(X_sparse_tran)
    assert_array_almost_equal(X_sparse.toarray(), X_sparse_tran_inv.toarray())


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_check_error(csc_container):
    X = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    X = csc_container(X)
    X_neg = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [-2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    X_neg = csc_container(X_neg)

    err_msg = (
        "The number of quantiles cannot be greater than "
        "the number of samples used. Got 1000 quantiles "
        "and 10 samples."
    )
    with pytest.raises(ValueError, match=err_msg):
        FastQuantileTransformer(subsample=10).fit(X)

    transformer = FastQuantileTransformer(n_quantiles=10)
    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit(X_neg)
    transformer.fit(X)
    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    with pytest.raises(ValueError, match=err_msg):
        transformer.transform(X_neg)

    X_bad_feat = np.transpose(
        [[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]]
    )
    err_msg = (
        "X has 2 features, but FastQuantileTransformer is expecting 3 features as input."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.inverse_transform(X_bad_feat)

    transformer = FastQuantileTransformer(n_quantiles=10).fit(X)
    # check that an error is raised if input is scalar
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        transformer.transform(10)
    # check that a warning is raised is n_quantiles > n_samples
    transformer = FastQuantileTransformer(n_quantiles=100)
    warn_msg = "n_quantiles is set to n_samples"
    with pytest.warns(UserWarning, match=warn_msg) as record:
        transformer.fit(X)
    assert len(record) == 1
    assert transformer.n_quantiles_ == X.shape[0]


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_sparse_ignore_zeros(csc_container):
    X = np.array([[0, 1], [0, 0], [0, 2], [0, 2], [0, 1]])
    X_sparse = csc_container(X)
    transformer = FastQuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)

    # dense case -> warning raise
    warning_message = (
        "'ignore_implicit_zeros' takes effect"
        " only with sparse matrix. This parameter has no"
        " effect."
    )
    with pytest.warns(UserWarning, match=warning_message):
        transformer.fit(X)

    X_expected = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.toarray())

    # consider the case where sparse entries are missing values and user-given
    # zeros are to be considered
    X_data = np.array([0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 0])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    X_sparse = csc_container((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    assert_almost_equal(X_expected, X_trans.toarray())

    transformer = FastQuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)
    X_data = np.array([-1, -1, 1, 0, 0, 0, 1, -1, 1])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6])
    X_sparse = csc_container((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array(
        [[0, 1], [0, 0.375], [0, 0.375], [0, 0.375], [0, 1], [0, 0], [0, 1]]
    )
    assert_almost_equal(X_expected, X_trans.toarray())
    assert_almost_equal(
        X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray()
    )

    # check in conjunction with subsampling
    transformer = FastQuantileTransformer(
        ignore_implicit_zeros=True, n_quantiles=5, subsample=8, random_state=0
    )
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.toarray())
    assert_almost_equal(
        X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray()
    )


def test_quantile_transform_dense_toy():
    X = np.array(
        [[0, 2, 2.6], [25, 4, 4.1], [50, 6, 2.3], [75, 8, 9.5], [100, 10, 0.1]]
    )

    transformer = FastQuantileTransformer(n_quantiles=5)
    transformer.fit(X)

    # using a uniform output, each entry of X should be map between 0 and 1
    # and equally spaced
    X_trans = transformer.fit_transform(X)
    X_expected = np.tile(np.linspace(0, 1, num=5), (3, 1)).T
    assert_almost_equal(np.sort(X_trans, axis=0), X_expected)

    X_test = np.array(
        [
            [-1, 1, 0],
            [101, 11, 10],
        ]
    )
    X_expected = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    assert_array_almost_equal(transformer.transform(X_test), X_expected)

    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)


def test_quantile_transform_subsampling():
    # Test that subsampling the input yield to a consistent results We check
    # that the computed quantiles are almost mapped to a [0, 1] vector where
    # values are equally spaced. The infinite norm is checked to be smaller
    # than a given threshold. This is repeated 5 times.

    # dense support
    n_samples = 1000000
    n_quantiles = 1000
    X = np.sort(np.random.sample((n_samples, 1)), axis=0)
    ROUND = 5
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = FastQuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 1e-2
        inf_norm_arr.append(inf_norm)
    # each random subsampling yield a unique approximation to the expected
    # linspace CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)

    # sparse support

    X = sparse.rand(n_samples, 1, density=0.99, format="csc", random_state=0)
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = FastQuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 1e-1
        inf_norm_arr.append(inf_norm)
    # each random subsampling yield a unique approximation to the expected
    # linspace CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)


def test_quantile_transform_subsampling_disabled():
    """Check the behaviour of `QuantileTransformer` when `subsample=None`."""
    X = np.random.RandomState(0).normal(size=(200, 1))

    n_quantiles = 5
    transformer = FastQuantileTransformer(n_quantiles=n_quantiles, subsample=None).fit(X)

    expected_references = np.linspace(0, 1, n_quantiles)
    assert_allclose(transformer.references_, expected_references)
    expected_quantiles = np.quantile(X.ravel(), expected_references)
    assert_allclose(transformer.quantiles_.ravel(), expected_quantiles)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_sparse_toy(csc_container):
    X = np.array(
        [
            [0.0, 2.0, 0.0],
            [25.0, 4.0, 0.0],
            [50.0, 0.0, 2.6],
            [0.0, 0.0, 4.1],
            [0.0, 6.0, 0.0],
            [0.0, 8.0, 0.0],
            [75.0, 0.0, 2.3],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 9.5],
            [100.0, 0.0, 0.1],
        ]
    )

    X = csc_container(X)

    transformer = FastQuantileTransformer(n_quantiles=10)
    transformer.fit(X)

    X_trans = transformer.fit_transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())

    transformer_dense = FastQuantileTransformer(n_quantiles=10).fit(X.toarray())

    X_trans = transformer_dense.transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    X_trans_inv = transformer_dense.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())


def test_quantile_transform_axis1():
    X = np.array([[0, 25, 50, 75, 100], [2, 4, 6, 8, 10], [2.6, 4.1, 2.3, 9.5, 0.1]])

    X_trans_a0 = quantile_transform(X.T, axis=0, n_quantiles=5)
    X_trans_a1 = quantile_transform(X, axis=1, n_quantiles=5)
    assert_array_almost_equal(X_trans_a0, X_trans_a1.T)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_bounds(csc_container):
    # Lower and upper bounds are manually mapped. We checked that in the case
    # of a constant feature and binary feature, the bounds are properly mapped.
    X_dense = np.array([[0, 0], [0, 0], [1, 0]])
    X_sparse = csc_container(X_dense)

    # check sparse and dense are consistent
    X_trans = FastQuantileTransformer(n_quantiles=3, random_state=0).fit_transform(X_dense)
    assert_array_almost_equal(X_trans, X_dense)
    X_trans_sp = FastQuantileTransformer(n_quantiles=3, random_state=0).fit_transform(
        X_sparse
    )
    assert_array_almost_equal(X_trans_sp.toarray(), X_dense)
    assert_array_almost_equal(X_trans, X_trans_sp.toarray())

    # check the consistency of the bounds by learning on 1 matrix
    # and transforming another
    X = np.array([[0, 1], [0, 0.5], [1, 0]])
    X1 = np.array([[0, 0.1], [0, 0.5], [1, 0.1]])
    transformer = FastQuantileTransformer(n_quantiles=3).fit(X)
    X_trans = transformer.transform(X1)
    assert_array_almost_equal(X_trans, X1)

    # check that values outside of the range learned will be mapped properly.
    X = np.random.random((1000, 1))
    transformer = FastQuantileTransformer()
    transformer.fit(X)
    assert transformer.transform([[-10]]) == transformer.transform([[np.min(X)]])
    assert transformer.transform([[10]]) == transformer.transform([[np.max(X)]])
    assert transformer.inverse_transform([[-10]]) == transformer.inverse_transform(
        [[np.min(transformer.references_)]]
    )
    assert transformer.inverse_transform([[10]]) == transformer.inverse_transform(
        [[np.max(transformer.references_)]]
    )


def test_quantile_transform_and_inverse():
    X_1 = iris.data
    X_2 = np.array([[0.0], [BOUNDS_THRESHOLD / 10], [1.5], [2], [3], [3], [4]])
    for X in [X_1, X_2]:
        transformer = FastQuantileTransformer(n_quantiles=1000, random_state=0)
        X_trans = transformer.fit_transform(X)
        X_trans_inv = transformer.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv, decimal=9)


def test_quantile_transform_nan():
    X = np.array([[np.nan, 0, 0, 1], [np.nan, np.nan, 0, 0.5], [np.nan, 1, 1, 0]])

    transformer = FastQuantileTransformer(n_quantiles=10, random_state=42)
    transformer.fit_transform(X)

    # check that the quantile of the first column is all NaN
    assert np.isnan(transformer.quantiles_[:, 0]).all()
    # all other column should not contain NaN
    assert not np.isnan(transformer.quantiles_[:, 1:]).any()


@pytest.mark.parametrize("array_type", ["array", "sparse"])
def test_quantile_transformer_sorted_quantiles(array_type):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15733
    # Taken from upstream bug report:
    # https://github.com/numpy/numpy/issues/14685
    X = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7] * 10)
    X = 0.1 * X.reshape(-1, 1)
    X = _convert_container(X, array_type)

    n_quantiles = 100
    qt = FastQuantileTransformer(n_quantiles=n_quantiles).fit(X)

    # Check that the estimated quantile thresholds are monotically
    # increasing:
    quantiles = qt.quantiles_[:, 0]
    assert len(quantiles) == 100
    assert all(np.diff(quantiles) >= 0)
