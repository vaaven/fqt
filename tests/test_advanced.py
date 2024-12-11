import numpy
import numpy as np
import torch
import pytest

from .context import *

n_samples = 10000
n_features = 10
ROUNDS = 5

def test_basic():
    tch = torch.ones(n_samples, n_features)
    transformer = FastQuantileTransformer(array_api_dispatch=False).fit(tch)
    result = transformer.transform(tch)
    assert type(result) == numpy.ndarray
    transformer = FastQuantileTransformer(array_api_dispatch=True).fit(tch)
    result = transformer.transform(tch)
    assert type(result) == torch.Tensor

def test_inplace():
    for _ in range(ROUNDS):
        tch = torch.rand(n_samples, n_features)
        tch_clone = tch.clone().detach()
        transformer = FastQuantileTransformer(array_api_dispatch=True).fit(tch)
        result = transformer.transform(tch)
        assert (tch == tch_clone).all()

def test_inverse():
    for _ in range(ROUNDS):
        tch = torch.rand(n_samples, n_features)
        transformer = FastQuantileTransformer(array_api_dispatch=True).fit(tch)
        result = transformer.transform(tch)
        assert torch.allclose(tch, transformer.inverse_transform(result), atol=1e-5)

def test_subsample():
    n_samples = 1000000
    n_quantiles = 1000
    X, _ = torch.sort(torch.rand(n_samples, 1), dim=0)
    inf_norm_arr = []
    for random_state in range(ROUNDS):
        transformer = FastQuantileTransformer(
            array_api_dispatch=True,
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
