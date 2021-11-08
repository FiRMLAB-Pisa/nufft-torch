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
# from typing import List, Tuple, Union

import numpy as np
import numba as nb

import torch
# from torch import Tensor


def prepare_nufft(coord, oversamp=1.125, width=3):
    pass

def nufft(input, coord, oversamp=1.25, width=4):
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
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.copy()

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    # Zero-pad
    output /= util.prod(input.shape[-ndim:])**0.5
    output = util.resize(output, os_shape)

    # FFT
    output = fft(output, axes=range(-ndim, 0), norm=None)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp)
    output = interp.interpolate(
        output, coord, kernel='kaiser_bessel', width=width, param=beta)
    output /= width**ndim

    return output

def nufft_adjoint(input, coord, oshape=None, oversamp=1.25, width=4):
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
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output = interp.gridding(input, coord, os_shape,
                             kernel='kaiser_bessel', width=width, param=beta)
    output /= width**ndim

    # IFFT
    output = ifft(output, axes=range(-ndim, 0), norm=None)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    return output


def toeplitz_psf(coord, shape, oversamp=1.25, width=4):
    """Toeplitz PSF for fast Normal non-uniform Fast Fourier Transform.
    While fast, this is more memory intensive.
    Args:
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        shape (tuple of ints): shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
            This is the shape of the input array of the forward nufft.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        array: PSF to be used by the normal operator defined in
            `sigpy.linop.NUFFT`
    See Also:
        :func:`sigpy.linop.NUFFT`
    """
    xp = backend.get_array_module(coord)
    with backend.get_device(coord):
        ndim = coord.shape[-1]

        new_shape = _get_oversamp_shape(shape, ndim, 2)
        new_coord = _scale_coord(coord, new_shape, 2)

        idx = [slice(None)]*len(new_shape)
        for k in range(-1, -(ndim + 1), -1):
            idx[k] = new_shape[k]//2

        d = xp.zeros(new_shape, dtype=xp.complex64)
        d[tuple(idx)] = 1

        psf = nufft(d, new_coord, oversamp, width)
        psf = nufft_adjoint(psf, new_coord, d.shape, oversamp, width)
        fft_axes = tuple(range(-1, -(ndim + 1), -1))
        psf = fft(psf, axes=fft_axes, norm=None) * (2**ndim)

        return psf

#%% utilities
def estimate_shape(coord):
    """Estimate array shape from coordinates.
    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.
    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    with backend.get_device(coord):
        shape = [int(coord[..., i].max() - coord[..., i].min())
                 for i in range(ndim)]

    return shape

def prod(shape):
    """Computes product of shape.
    
    Args:
        shape (tuple or list): shape.
        
    Returns:
        Product.
    """
    return np.prod(shape, dtype=np.int64)

def resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.
    
    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.
        
    Returns:
        array: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    xp = backend.get_array_module(input)
    output = xp.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))

def fft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """FFT function that supports centering.
    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute the FFT.
        norm (Nonr or ``"ortho"``): Keyword to specify the normalization mode.
    Returns:
        array: FFT result of dimension oshape.
    See Also:
        :func:`numpy.fft.fftn`
    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.fftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype,
                     np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype, copy=False)

    return output


def ifft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.
    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute
            the inverse FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.
    Returns:
        array of dimension oshape.
    See Also:
        :func:`numpy.fft.ifftn`
    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.ifftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype,
                     np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype)

    return output

def _fftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    xp = backend.get_array_module(input)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)
    tmp = xp.fft.ifftshift(tmp, axes=axes)
    tmp = xp.fft.fftn(tmp, axes=axes, norm=norm)
    output = xp.fft.fftshift(tmp, axes=axes)
    return output


def _ifftc(input, oshape=None, axes=None, norm='ortho'):
    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    xp = backend.get_array_module(input)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)
    tmp = xp.fft.ifftshift(tmp, axes=axes)
    tmp = xp.fft.ifftn(tmp, axes=axes, norm=norm)
    output = xp.fft.fftshift(tmp, axes=axes)
    return output


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.copy()
    for i in range(-ndim, 0):
        scale = ceil(oversamp * shape[i]) / shape[i]
        shift = ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i) for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta):
    xp = backend.get_array_module(input)
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)
        idx = xp.arange(i, dtype=output.dtype)

        # Calculate apodization
        apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
        apod /= xp.sinh(apod)
        output *= apod.reshape([i] + [1] * (-a - 1))

    return output