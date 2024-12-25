
# Fast Quantile Transformer

[![PyPI - Version](https://img.shields.io/pypi/v/fqt.svg)](https://pypi.org/project/fqt)

Improved version of Quantile Transfomer from scikit-learn package which provides ability to work with torch tensors on CUDA and other minor improvements.

## Documentation

> [!WARNING]
> Note that FastQuantileTransformer may not work on scikit-learn version 1.6.0 with numpy.ndarray. It is caused by this [bug](https://github.com/scikit-learn/scikit-learn/issues/29107).

Current version provides same features as the original one, but also three more atributes which are ``` array_api_dispatch, replace_subsamples, noise_distribution ```.

* `array_api_dispatch: bool, default is False`, while False converts all array_like object to numpy.ndarray or numpy.matrix (depends on sparse property). For example:

```python
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> type(FastQuantileTransformer().fit_transform(a))
numpy.ndarray
```

However, if array_like is set True, then transformer will work with object in their own namespace:

```python
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> type(FastQuantileTransformer(array_api_dispatch=True).fit_transform(a))
torch.Tensor
```
> [!NOTE]
> Due to unavailability of quantile function at array api standart, it is guaranted to work properly only with numpy, scipy and torch.

*  `replace_subsamples: bool, default is True`, specify the policy of samping subsample for quantile selection. If set to True, some objects may occur multiple times in subsample. Otherwise, if set to False, one object may occur only one time in subsample.  

***QuantileTrensformer from sklearn by default follows array_api_dispatch=False strategy***. However, it seems overkilling, because it makes noticable effect only when subsample almost equal n_samples, but takes a lot of time (check _Benchmark_ section). 

- `noise_distribution: Optional[Literal['uniform']], default is None`, describes the noise property that is added to the data: 
    - `normal` is normal distribution with `loc=0.0, scale=1e-5`
    - `uniform` is uniform distribution in range `[-1e-5, 1e-5]`

Quantile transform from sklearn performs interpolations among quantiles twice due to case of repeating values (however, it's not actually helping, see _add link_), but when you're using noise provided in FastQuantileTransformer it's absolutely useless. That's why FastQuantileTransformer with noise works noticably faster even on numpy.ndarray.


By paying attention to the description of the parameters you can note that `FastQuantileTransformer(replace_subsamples=False)` is absolutely the same as `QuantileTrensformer()`



## Benchmarks

**TODO**