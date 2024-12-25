import torch
from sklearn.utils._array_api import get_namespace, get_namespace_and_device

def interpolate_torch(x, Xp, fp, side='left', left=None, right=None):
    """
    Perform linear interpolation in 1D using PyTorch.

    Parameters:
    x : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values.
    xp : torch.Tensor
        The x-coordinates of the data points.
    fp : torch.Tensor
        The y-coordinates of the data points.
    left : float, optional
        Value to return for x < xp[0]. If not provided, it will return the first value of fp.
    right : float, optional
        Value to return for x > xp[-1]. If not provided, it will return the last value of fp.

    Returns:
    torch.Tensor
        The interpolated values at x.
    """
    xp, _ = get_namespace(x)

    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    idx = xp.searchsorted(Xp, x, side=side)

    result = xp.empty_like(x)

    mask_left = x < Xp[0]
    result[mask_left] = left

    mask_right = x > Xp[-1]
    result[mask_right] = right

    mask_within = ~mask_left & ~mask_right
    x_within = x[mask_within]
    idx_within = idx[mask_within]

    x0 = Xp[idx_within - 1]
    x1 = Xp[idx_within]
    f0 = fp[idx_within - 1]
    f1 = fp[idx_within]

    result[mask_within] = f0 + (f1 - f0) * (x_within - x0) / (x1 - x0)

    return result
