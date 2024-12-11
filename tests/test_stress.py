import torch
import pytest
import numpy as np
from sklearn.preprocessing import QuantileTransformer

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
        torch_result = FastQuantileTransformer(array_api_dispatch=True, random_state=42).fit_transform(torch_data)
        assert np.allclose(np_result, torch_result.numpy())

def test_basic_normal():
    for _ in range(ROUNDS):
        np_data = rng.rand(n_samples, n_features)
        np_result = QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(np_data)
        torch_data = torch.asarray(np_data)
        torch_result = FastQuantileTransformer(array_api_dispatch=True, output_distribution='normal', random_state=42).fit_transform(torch_data)
        assert np.allclose(np_result, torch_result.numpy())