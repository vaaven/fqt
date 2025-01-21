
# Fast Quantile Transformer

[![PyPI - Version](https://img.shields.io/pypi/v/fqt.svg)](https://pypi.org/project/fqt)

Most of the code from this project was taken and modified from scikit-learn package. Credits to [sklearn](https://github.com/scikit-learn/scikit-learn).

Improved version of Quantile Transfomer from scikit-learn package which provides ability to work with torch tensors on CUDA and other minor improvements.

## Documentation

> [!WARNING]
> Note that FastQuantileTransformer may not work on scikit-learn version 1.6.0 with numpy.ndarray. It is caused by this [bug](https://github.com/scikit-learn/scikit-learn/issues/29107).

Current version provides same features as the original one, but also three more atributes which are ``` replace_subsamples, noise_distribution, epsilon ```.

By default, Fast Quantile Transformer works almost the same as Quantile Transformer from sklearn:

```python
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> type(FastQuantileTransformer().fit_transform(a))
<class 'numpy.ndarray'>
```

However, if array_api_dispatch is set True, then transformer will work with object in their own namespace:

```python
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> sklearn.set_config(array_api_dispatch=True)
>>> type(FastQuantileTransformer().fit_transform(a))
<class 'torch.Tensor'>
```
> [!NOTE]
> Due to unavailability of quantile function at array api standart, it is guaranted to work properly only with numpy, scipy and torch.

*  `replace_subsamples: bool, default is True`, specify the policy of samping subsample for quantile selection. If set to True, some objects may occur multiple times in subsample. Otherwise, if set to False, one object may occur only one time in subsample.  

***QuantileTrensformer from sklearn by default follows replace_subsamples=False strategy***. However, it seems overkilling, because it makes noticable effect only when subsample almost equal n_samples, but takes a lot of time (check _Benchmark_ section). 

- `noise_distribution: Optional[Literal['uniform']], default is None`, describes the noise property that is added to the data: 
    - `normal` is normal distribution with `loc=0.0, scale=epsilon`
    - `uniform` is uniform distribution in range `[-epsilon, epsilon]`
    - `deterministic`, changes the distribution of border values. For more details see `examples/noise_policy.ipynb`.

By default `epsilon=1e-5`. However, you can specify it in arguments of FastQuantileTransformer.

Quantile transform from sklearn performs interpolations among quantiles twice due to case of repeating values (however, it's not actually helping, see _add link_), but when you're using noise provided in FastQuantileTransformer it's absolutely useless. That's why FastQuantileTransformer with noise works noticably faster even on numpy.ndarray.

> [!NOTE]
> By paying attention to the description of the parameters you can note that `FastQuantileTransformer(replace_subsamples=False)` is absolutely the same as `QuantileTrensformer()`


## Benchmarks

All the tests are performed with array of 1000000 samples and 100 features. For more details please pay attention to examples/benchmarks.ipynb

All the tests are performed using kaggle settings with single GPU T4.

| attributes\transformer       | sklearn | fqt numpy | fqt torch | fqt torch + cuda |
|------------------------------|---------|-----------|-----------|------------------|
| `standard`                   | 22.3s   | 22.2s     | 22.3s     | 3.7s             |
| `subsamples=100000`          | 23.1s   | 23.0s     | 23.2s     | 3.7s             |
|`noise_distribution='uniform'`| -       | 14.5s     | 14.6s     | 4.7s             |