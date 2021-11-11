# -*- coding: utf-8 -*-
"""
Utilities for data manipulation.

Contains utilities for data handling in NUFFT (i.e. padding/cropping and scaling).
"""
from typing import Union, Tuple, List

import numpy as np
import numba as nb

import torch
from torch import Tensor


def prod(shape: Union[List, Tuple]) -> np.int64:
    """Computes product of shape.

    Args:
        shape (tuple or list): shape.

    Returns:
        Product.
    """
    return np.prod(shape, dtype=np.int64)


def resize(data_in: Tensor,
           oshape: Union[List[int], Tuple[int]],
           ishift: Union[List[int], Tuple[int], None] = None,
           oshift: Union[List[int], Tuple[int], None] = None) -> Tensor:
    """Resize with zero-padding or cropping.

    Args:
        data_in (tensor): Input Tensor.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.

    Returns:
        tensor: Zero-padded or cropped result.
    """
    ishape1, oshape1 = _expand_shapes(data_in.shape, oshape)

    if ishape1 == oshape1:
        return data_in.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    data_out = torch.zeros(  # pylint: disable=no-member
        oshape1, dtype=data_in.dtype)
    data_in = data_in.reshape(ishape1)
    data_out[oslice] = data_in[islice]

    return data_out.reshape(oshape)


def scale_coord(coord: Tensor,
                shape: Union[List[int], Tuple[int]],
                oversamp: float) -> Tensor:
    """Scale coordinates to fit oversampled grid.

    Args:
        coord (tensor): Coordinate tensor of shape [nframes, ..., ndim]
        shape (list or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.

    Returns:
        tensor: Scaled coordinates of shape [nframes, ..., ndim]
        
    """
    ndim = coord.shape[-1]
    output = coord.clone()
    for i in range(-ndim, 0):
        scale = np.ceil(oversamp * shape[i]) / shape[i]
        shift = np.ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def get_oversamp_shape(shape: Union[List[int], Tuple[int]],
                       oversamp: float,
                       ndim: int) -> List[int]:
    """Computes size of oversampled grid for given oversampling factor.

    Args:
        shape (list or tuple of ints): Cartesian grid size.
        ndim (int): Number of grid spatial dimensions.
        oversamp (float): Grid oversampling factor.

    Returns:
        list: shape of oversampled grid.
    """
    return list(shape)[:-ndim] + [np.ceil(oversamp * i).astype(np.int16) for i in shape[-ndim:]]


def apodize(data_in: Tensor,
            ndim: int,
            oversamp: float,
            width: Union[List[int], Tuple[int]],
            beta: Union[List[float], Tuple[float]]) -> Tensor:
    """
    Apodize data in image space to remove effect of kernel convolution.

    Args
        data_in (tensor): input image space data.
        ndim (int):  Number of grid spatial dimensions.
        oversamp (float): Grid oversampling factor.
        width (list or tuple of int): Interpolation kernel full-width.
        beta (list or tuple of floats): Kaiser-Bessel beta parameter.

    Returns:
        data_out (tensor): apodized data.
    """
    data_out = data_in
    for axis in range(-ndim, 0):
        i = data_out.shape[axis]
        os_i = np.ceil(oversamp * i)
        idx = torch.arange(  # pylint: disable=no-member
            i, dtype=torch.float32)

        # Calculate apodization
        apod = (beta[axis]**2 - (np.pi * width[axis] * (idx - i // 2) / os_i)**2)**0.5
        apod /= torch.sinh(apod)  # pylint: disable=no-member
        data_out *= apod.reshape([i] + [1] * (-axis - 1))

    return data_out


def numba2pytorch(array, requires_grad=True):  # pragma: no cover
    """Zero-copy conversion from Numpy/Numba CUDAarray to PyTorch tensor.

    Args:
        array (numpy/cupy array): input.
        requires_grad(bool): Set .requires_grad output tensor

    Returns:
        PyTorch tensor.
    """
    if torch.cuda.is_available() is True:
        if nb.cuda.is_cuda_array(array) is True:
            tensor = torch.as_tensor(  # pylint: disable=no-member
                array, device="cuda")  # pylint: disable=no-member
        else:
            tensor = torch.from_numpy(array)  # pylint: disable=no-member
    else:
        tensor = torch.from_numpy(array)  # pylint: disable=no-member

    tensor.requires_grad = requires_grad
    return tensor.contiguous()


def pytorch2numba(tensor):  # pragma: no cover
    """Zero-copy conversion from PyTorch tensor to Numpy/Numba CUDA array.

    Args:
        tensor (PyTorch tensor): input.

    Returns:
        Numpy/Numba CUDA array.

    """
    device = tensor.device
    if device.type == 'cpu':
        array = tensor.detach().contiguous().numpy()
    else:
        array = nb.cuda.as_cuda_array(tensor.contiguous())

    return array


def _expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)


def normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))
