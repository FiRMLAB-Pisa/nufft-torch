# -*- coding: utf-8 -*-
"""
Non-uniform Fourier Transform subroutines.

Uses interpolation routines from _interp.py module.

Adapted from SigPy [1]. Compared to [1], we use a reduced memory
radix tensor decomposition of pre-computed interpolators, to speed-up
computation with respect to precomputation-free NUFFT [2,3] while
mantaining a reasonably light memory footprint in high-dimensional
problems (non-separable 3D+t data). We include interpolation
routines with embedded low-rank subspace projection operators for
model-based MRI reconstructions.

[1]: Ong, F., and M. Lustig. "SigPy: a python package for high performance iterative reconstruction."
         Proceedings of the ISMRM 27th Annual Meeting, Montreal, Quebec, Canada. Vol. 4819. 2019.
[2]: Lin, Jyh-Miin. "Python non-uniform fast Fourier transform (PyNUFFT):
     An accelerated non-Cartesian MRI package on a heterogeneous platform (CPU/GPU)."
     Journal of Imaging 4.3 (2018): 51.
[3]: Lin, Jyh-Miin, et al.
     "Memory reduced non-Cartesian MRI encoding using the mixed-radix tensor product on CPU and GPU."
     arXiv preprint arXiv:1903.08365 (2019).
[4]: McGivney DF, Pierre E, Ma D, et al.
     SVD compression for magnetic resonance fingerprinting in the time domain.
     IEEE Trans Med Imaging. 2014;33(12):2311-2322. doi:10.1109/TMI.2014.2337321

"""
from typing import List, Tuple, Dict, Union

import numpy as np

import torch
from torch import Tensor

from src import _util, _interp


def prepare_nufft(coord: Tensor,
                  shape: Union[int, List[int, ...], Tuple[int, ...]],
                  oversamp: Union[float, List[float, ...], Tuple[float, ...]] = 1.125,
                  width: Union[int, List[int, ...], Tuple[int, ...]] = 3,
                  basis: Union[None, Tensor] = None,
                  device: str = 'cpu') -> Dict:
    """Precompute nufft object for faster t_nufft / t_nufft_adjoint.
    Args:
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        shape (int or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.

    Returns:
        dict: structure containing interpolator object.
    """
    # get parameters
    ndim = coord.shape[-1]

    if np.isscalar(width):
        width = torch.tensor([width] * ndim, dtype=torch.int) # pylint: disable=no-member
    else:
        width = torch.tensor(width, dtype=torch.int) # pylint: disable=no-member

    # calculate Kaiser-Bessel beta parameter
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    if np.isscalar(shape):
        shape = torch.tensor([shape] * ndim, dtype=torch.int) # pylint: disable=no-member
    else:
        shape = torch.tensor(shape, dtype=torch.int) # pylint: disable=no-member

    os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

    # adjust coordinates
    coord = _util.scale_coord(coord, shape, oversamp)

    # compute interpolator
    sparse_coeff = _interp.prepare_interpolator(
        coord, os_shape, width, beta, device)

    return {'ndim': ndim,
            'oversamp': oversamp,
            'width': width,
            'beta': beta,
            'os_shape': os_shape,
            'oshape': shape,
            'sparse_coeff': sparse_coeff,
            'device': device}


def nufft(image: Tensor, interpolator: Dict, adjoint_basis: Union[None, Tensor] = None) -> Tensor:
    """Non-uniform Fast Fourier Transform.
    Args:
        input (array): input signal domain array of shape
            (..., n_{ndim - 1}, ..., n_1, n_0),
            where ndim is specified by coord.shape[-1]. The nufft
            is applied on the last ndim axes, and looped over
            the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        array: Fourier domain data of shape
            input.shape[:-ndim] + coord.shape[:-1].
    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    os_shape = interpolator['os_shape']
    sparse_coeff = interpolator['sparse_coeff']

    # copy input to avoid original data modification
    image = image.clone()

    # Apodize
    _util.apodize(image, ndim, oversamp, width, beta)

    # Zero-pad
    image /= _util.prod(image.shape[-ndim:])**0.5
    image = _util.resize(image, os_shape)

    # FFT
    kdata = fft(image, axes=range(-ndim, 0), norm=None)

    # Interpolate
    kdata = _interp.interpolate(kdata, sparse_coeff, adjoint_basis)  # pylint: disable=no-value-for-parameter
    kdata /= _util.prod(width)

    return kdata


def nufft_adjoint(kdata: Tensor, interpolator: Dict, basis: Union[None, Tensor] = None) -> Tensor:
    """Adjoint non-uniform Fast Fourier Transform.
    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        array: signal domain array with shape specified by oshape.
    See Also:
        :func:`sigpy.nufft.nufft`
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    os_shape = interpolator['os_shape']
    oshape = interpolator['oshape']
    sparse_coeff = interpolator['sparse_coeff']

    # Gridding
    kdata = _interp.gridding(kdata, sparse_coeff, basis) # pylint: disable=no-value-for-parameter
    kdata /= _util.prod(width)

    # IFFT
    image = ifft(kdata, axes=range(-ndim, 0), norm=None)

    # Crop
    image = _util.resize(image, oshape)
    image *= _util.prod(os_shape[-ndim:]) / _util.prod(oshape[-ndim:])**0.5

    # Apodize
    _util.apodize(image, ndim, oversamp, width, beta)

    return image


def prepare_toeplitz(coord: Tensor,
                     shape: Union[int, List[int, ...], Tuple[int, ...]],
                     oversamp: Union[float, List[float, ...], Tuple[float, ...]] = 1.125,
                     width: Union[int, List[int, ...], Tuple[int, ...]] = 3,
                     device: str = 'cpu',
                     basis: Tensor = None,
                     dcf: Tensor = None) -> Dict:

    # get initial data shape
    nframes = coord.shape[0]
    pts_shape = coord.shape[1:-1]
    npts = _util.prod(pts_shape)
    ndim = coord.shape[-1]

    # reshape input
    coord = coord.reshape([nframes, npts, ndim])

    if np.isscalar(width):
        width = torch.tensor([width] * ndim, dtype=torch.int) # pylint: disable=no-member
    else:
        width = torch.tensor(width, dtype=torch.int) # pylint: disable=no-member

    # calculate Kaiser-Bessel beta parameter
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    if np.isscalar(shape):
        shape = torch.tensor([shape] * ndim, dtype=torch.int) # pylint: disable=no-member
    else:
        shape = torch.tensor(shape, dtype=torch.int) # pylint: disable=no-member

    # get oversampled grid
    os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

    # scale coordinates
    coord = _util.scale_coord(coord, shape, oversamp)

    # actual kernel precomputation
    kernel = _interp.prepare_toeplitz(coord,
                                      os_shape,
                                      width=width,
                                      beta=beta,
                                      device=device,
                                      basis=basis,
                                      dcf=dcf)

    return {'kernel': kernel, 'oversamp': oversamp, 'ndim': ndim}


def nufft_selfadjoint(data_in: Tensor, toeplitz: Dict) -> Tensor:
    """Perform in-place fast self-adjoint by convolution with spatio-temporal kernel matrix.

    Args:
        data_in (tensor): Input image.
        toeplitz (dict): Pre-computed spatio-temporal kernel.

    Returns
        data_out (tensor): Output image.

    """
    # unpack input
    toeplitz_kernel = toeplitz['kernel']
    oversamp = toeplitz['oversamp']
    ndim = toeplitz['ndim']
    islowrank = toeplitz_kernel['islowrank']

    ishape = list(data_in.shape)
    data_in = data_in.reshape(
        [ishape[0], _util.prod(ishape[1:-ndim]), *ishape[-ndim:]])

    # Get oversampled shape
    shape = list(input.shape)
    os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

    # Zero-pad
    data_in = _util.resize(data_in, os_shape)

    # FFT
    data_in = fft(data_in, axes=range(-ndim, 0), center=False, norm=None)

    # Reshape input
    if islowrank is True:
        data_in = data_in.reshape(
            [shape[0], shape[1], _util.prod(os_shape[-ndim:])])

    # Preallocate output
    data_out = torch.zeros( # pylint: disable=no-member
        data_in.shape, dtype=data_in.dtype, device=data_in.device)

    # Perform convolution
    _interp.toeplitz(data_out, data_in, toeplitz_kernel)

    # Reshape output
    if islowrank is True:
        data_out = data_out.reshape(os_shape)

    # IFFT
    data_out = ifft(data_out, axes=range(-ndim, 0), center=False, norm=None)

    # Crop
    data_out = _util.resize(data_out, shape)
    data_out = data_out.reshape(ishape)

    if islowrank is True:
        data_out = data_out[:, 0]

    return data_out


def fft(data_in, oshape=None, axes=None, center=True, norm='ortho'):
    """FFT function that supports centering.
    Args:
        data_in (torch): input tensor.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute the FFT.
        norm (Nonr or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        tensor: FFT result of dimension oshape.

    See Also:
        :func:`torch.fft.fftn`
    """
    # allow for single scalar axis
    if np.isscalar(axes):
        axes = [axes]

    # transform to list to allow range object
    axes = list(axes)

    if center:
        data_out = _fftc(data_in, oshape=oshape, axes=axes, norm=norm)
    else:
        data_out = torch.fft.fftn(data_in, n=oshape, dim=axes, norm=norm)

    if torch.is_complex(data_in) and data_in.dtype != data_out.dtype: # pylint: disable=no-member
        data_out = data_out.to(data_in.dtype, copy=False)

    return data_out


def ifft(data_in, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.
    Args:
        data_in (tensor): input tensor.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute
            the inverse FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        tensor of dimension oshape.

    See Also:
        :func:`torch.fft.ifftn`
    """
    # allow for single scalar axis
    if np.isscalar(axes):
        axes = [axes]

    # transform to list to allow range object
    axes = list(axes)

    if center:
        data_out = _fftc(data_in, oshape=oshape, axes=axes, norm=norm)
    else:
        data_out = torch.fft.ifftn(data_in, n=oshape, dim=axes, norm=norm)

    if torch.is_complex(data_in) and data_in.dtype != data_out.dtype: # pylint: disable=no-member
        data_out = data_out.to(data_in.dtype, copy=False)

    return data_out


def _fftc(data_in, oshape=None, axes=None, norm='ortho'):

    ndim = data_in.ndim
    axes = _util.normalize_axes(axes, ndim)

    if oshape is None:
        oshape = data_in.shape

    tmp = _util.resize(data_in, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
    data_out = torch.fft.fftshift(tmp, dim=axes)

    return data_out


def _ifftc(data_in, oshape=None, axes=None, norm='ortho'):
    ndim = data_in.ndim
    axes = _util.normalize_axes(axes, ndim)

    if oshape is None:
        oshape = data_in.shape

    tmp = _util.resize(data_in, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
    data_out = torch.fft.fftshift(tmp, dim=axes)

    return data_out
