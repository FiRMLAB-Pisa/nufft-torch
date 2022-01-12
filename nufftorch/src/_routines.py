# -*- coding: utf-8 -*-
"""
Main FFT/NUFFT/Toeplitz routines.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=redefined-builtin
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=protected-access
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments

from typing import List, Tuple, Dict, Union

import numpy as np
import torch

from torch import Tensor

from nufftorch.src._subroutines import (Apodize,
                                        Crop,
                                        DeGrid,
                                        DeviceDispatch,
                                        FFT,
                                        Grid,
                                        IFFT,
                                        Toeplitz,
                                        ZeroPad)

from nufftorch.src._factory import NUFFTFactory, NonCartesianToeplitzFactory


def prepare_nufft(coord: Tensor,
                  shape: Union[int, List[int], Tuple[int]],
                  oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                  width: Union[int, List[int], Tuple[int]] = 3,
                  basis: Union[None, Tensor] = None,
                  sharing_width: Union[None, int] = None,
                  device: Union[str, torch.device] = 'cpu',
                  threadsperblock: int = 512) -> Dict:
    """Precompute NUFFT coefficients."""
    return NUFFTFactory()(coord, shape, width, oversamp, basis, sharing_width, device, threadsperblock)


def nufft(image: Tensor, interpolator: Dict) -> Tensor:
    """Non-uniform Fast Fourier Transform."""
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    kernel_dict = interpolator['kernel_dict']
    device_dict = interpolator['device_dict']
    device = device_dict['device']

    # Copy input to avoid original data modification
    image = image.clone()

    # Offload to computational device
    dispatcher = DeviceDispatch(
        computation_device=device, data_device=image.device)
    image = dispatcher.dispatch(image)

    # Apodize
    Apodize(image.shape[-ndim:], oversamp, width, beta, device)(image)

    # Zero-pad
    image = ZeroPad(oversamp, image.shape[-ndim:])(image)

    # FFT
    kdata = FFT()(image, axes=range(-ndim, 0), norm='ortho')

    # Interpolate
    kdata = DeGrid(device_dict)(kdata, kernel_dict)

    # Bring back to original device
    kdata, image = dispatcher.gather(kdata, image)

    return kdata


def nufft_adjoint(kdata: Tensor, interpolator: Dict) -> Tensor:
    """Adjoint Non-uniform Fast Fourier Transform."""
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    kernel_dict = interpolator['kernel_dict']
    device_dict = interpolator['device_dict']
    device = device_dict['device']

    # Offload to computational device
    dispatcher = DeviceDispatch(
        computation_device=device, data_device=kdata.device)
    kdata = dispatcher.dispatch(device)

    # Gridding
    kdata = Grid(device_dict)(kdata, kernel_dict)

    # IFFT
    image = IFFT()(kdata, axes=range(-ndim, 0), norm='ortho')

    # Crop
    image = Crop(oversamp, image.shape[:-ndim])(image)

    # Apodize
    Apodize(ndim, oversamp, width, beta, device)(image)

    # Bring back to original device
    image, kdata = dispatcher.gather(image, kdata)

    return image


def prepare_noncartesian_toeplitz(coord: Tensor,
                                  shape: Union[int, List[int], Tuple[int]],
                                  prep_oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                                  comp_oversamp: Union[float, List[float], Tuple[float]] = 1.0,
                                  width: Union[int, List[int], Tuple[int]] = 3,
                                  basis: Union[Tensor, None] = None,
                                  sharing_width: Union[int, None] = None,
                                  device: Union[str, torch.device] = 'cpu',
                                  threadsperblock: int = 512,
                                  dcf: Union[Tensor, None] = None) -> Dict:
    """Prepare Toeplitz matrix for Non-Cartesian sampling."""
    return NonCartesianToeplitzFactory()(coord, shape, prep_oversamp, comp_oversamp, width,
                                         basis, sharing_width, device, threadsperblock, dcf)


def toeplitz_convolution(image: Tensor, toeplitz_dict: Dict) -> Tensor:
    """Perform in-place fast self-adjoint by convolution with spatio-temporal kernel matrix.

    Args:
        data_in (tensor): Input image.
        toeplitz (dict): Pre-computed spatio-temporal kernel.

    Returns
        data_out (tensor): Output image.

    """
    # unpack input
    mtf = toeplitz_dict['mtf']
    islowrank = toeplitz_dict['islowrank']
    device_dict = toeplitz_dict['device_dict']
    device = device_dict['device']
    ndim = toeplitz_dict['ndim']
    oversamp = toeplitz_dict['oversamp']

    # Offload to computational device
    dispatcher = DeviceDispatch(
        computation_device=device, data_device=image.device)
    image = dispatcher.dispatch(image)

    # reshape for computation
    ishape = list(image.shape)
    image = image.reshape(ishape[0], np.prod(ishape[1:-ndim]), *ishape[-ndim:])

    # Zero-pad
    image = ZeroPad(oversamp, image.shape[-ndim:])(image)
    os_shape = image.shape

    # FFT
    kdata_in = FFT()(image, axes=range(-ndim, 0), norm='ortho')

    # Reshape input
    if islowrank is True:
        kdata_in = kdata_in.reshape(
            *kdata_in.shape[:2], np.prod(kdata_in.shape[-ndim:]))

    # Preallocate output
    kdata_out = torch.zeros(
        kdata_in.shape, dtype=kdata_in.dtype, device=kdata_in.device)

    # Perform convolution
    Toeplitz(kdata_in.size, device_dict)(kdata_out, kdata_in, mtf)

    # Reshape output
    if islowrank is True:
        kdata_out = kdata_out.reshape(os_shape)

    # IFFT
    image = IFFT()(kdata_out, axes=range(-ndim, 0), norm='ortho')

    # Crop
    image = Crop(oversamp, image.shape[:-ndim])(image)

    # Bring back to original device
    image = dispatcher.gather(image)

    return image
