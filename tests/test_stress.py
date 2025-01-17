import torch
import pytest
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import random
import sklearn

from .context import *

ROUNDS = 100
n_features = 10
n_samples = 10000
rng = np.random.RandomState(0)


def test_basic_uniform():
    for _ in range(ROUNDS):
        np_data = rng.rand(n_samples, n_features)
        np_result = QuantileTransformer(random_state=42).fit_transform(np_data)
        torch_data = torch.asarray(np_data)
        with sklearn.config_context(array_api_dispatch=True):
            transformer = FastQuantileTransformer(random_state=42)
            torch_result = transformer.fit_transform(torch_data)
        assert np.allclose(np_result, torch_result.numpy())
        assert np.allclose(np_data, transformer.inverse_transform(torch_result), atol=1e-3)


def test_basic_normal():
    for _ in range(ROUNDS):
        np_data = rng.rand(n_samples, n_features)
        np_result = QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(np_data)
        torch_data = torch.asarray(np_data)
        with sklearn.config_context(array_api_dispatch=True):
            transformer = FastQuantileTransformer(output_distribution='normal', random_state=42)
            torch_result = transformer.fit_transform(torch_data)
        assert np.allclose(np_result, torch_result.numpy())
        assert np.allclose(np_data, transformer.inverse_transform(torch_result), atol=1e-3)


def test_subsample():
    n_samples = 100000
    n_subsamples = 10000
    for _ in range(ROUNDS // 5):
        np_data = rng.rand(n_samples, n_features)
        np_result = QuantileTransformer(subsample=n_subsamples, n_quantiles=n_subsamples // 10,
                                        random_state=42).fit_transform(np_data)
        torch_data = torch.asarray(np_data)
        with sklearn.config_context(array_api_dispatch=True):
            transformer = FastQuantileTransformer(subsample=n_subsamples, n_quantiles=n_subsamples // 10, random_state=42)
            torch_result = transformer.fit_transform(torch_data)
            assert np.allclose(torch_data, transformer.inverse_transform(torch_result), atol=1e-3)


def test_random_state():
    for _ in range(ROUNDS // 100):
        np_data = rng.rand(n_samples, n_features)
        torch_data = torch.asarray(np_data)
        true_result = None
        for __ in range(100):
            with sklearn.config_context(array_api_dispatch=True):
                transformer = FastQuantileTransformer(output_distribution='normal',
                                                    random_state=42, n_quantiles=2, subsample=100)
                torch_result = transformer.fit_transform(torch_data)
            if true_result is None:
                true_result = torch_result
            assert torch.allclose(torch_result, true_result)
