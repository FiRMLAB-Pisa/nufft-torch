# -*- coding: utf-8 -*-
"""
Main FFT/NUFFT/Toeplitz routines.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=redefined-builtin

from typing import List, Tuple, Dict, Union, ModuleType

import numpy as np
import numba as nb

import torch
from torch import Tensor, device

from lr_nufft_torch.src import _cpu

if torch.cuda.is_available():
    is_cuda_enabled = True
    from lr_nufft_torch.src import _cuda

else:
    is_cuda_enabled = False


def fft():
    """Cartesian FFT."""
    pass


def ifft():
    """Cartesian Inverse FFT."""
    pass


def prepare_nufft(coord: Tensor,
                  shape: Union[int, List[int], Tuple[int]],
                  oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                  width: Union[int, List[int], Tuple[int]] = 3,
                  basis: Union[None, Tensor] = None,
                  sharing_width: Union[None, int] = None,
                  device: Union[str, device] = 'cpu') -> Dict:
    """Precompute nufft object for faster t_nufft / t_nufft_adjoint.
    Args:
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        shape (int or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.
        sharing_width (int): Width of sharing window in sliding-window reconstruction.
        device: CPU or CUDA.

    Returns:
        dict: structure containing interpolator object.
    """
    return NUFFTFactory()(coord, shape, width, oversamp, basis, sharing_width, device)


def nufft(image: Tensor, interpolator: Dict) -> Tensor:
    """Non-uniform Fast Fourier Transform.
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    kernel_dict = interpolator['kernel_dict']
    device = interpolator['device']

    # Copy input to avoid original data modification
    image = image.clone()

    # Offload to computational device
    dispatcher = DeviceDispatch(
        computation_device=device, data_device=image.device)
    image = dispatcher.dispatch(image)

    # Apodize
    Apodize(ndim, oversamp, width, beta, device)(image)

    # Zero-pad
    image = ZeroPad(oversamp, image.shape[-ndim:])(image)

    # FFT
    kdata = FFT()(image, axes=range(-ndim, 0), norm='ortho')

    # Interpolate
    kdata = DeGrid(device)(image, kernel_dict)

    # Bring back to original device
    kdata, image = dispatcher.gather(kdata, image)

    return kdata


def nufft_adjoint(kdata: Tensor, interpolator: Dict) -> Tensor:
    """Adjoint Non-uniform Fast Fourier Transform.
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    kernel_dict = interpolator['kernel_dict']
    device = interpolator['device']

    # Offload to computational device
    dispatcher = DeviceDispatch(
        computation_device=device, data_device=kdata.device)
    kdata = dispatcher.dispatch(device)

    # Gridding
    kdata = Grid(device)(kdata, kernel_dict)

    # IFFT
    image = IFFT()(kdata, axes=range(-ndim, 0), norm='ortho')

    # Crop
    image = Crop(oversamp, image.shape[:-ndim])(image)

    # Apodize
    Apodize(ndim, oversamp, width, beta, device)(image)

    # Bring back to original device
    image, kdata = dispatcher.gather(image, kdata)

    return image

