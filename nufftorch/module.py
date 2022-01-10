# -*- coding: utf-8 -*-
"""
Object-oriented wrappers for Non-uniform Fourier Transform routines.


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

import torch
from torch import Tensor

from lr_nufft_torch import _fourier


class AbstractNUFFT(torch.nn.Module):  # pylint: disable=abstract-method
    """ common class for forward and adjoint NUFFT. """

    interpolator: Dict = {}

    def __init__(self,
                 coord: Union[None, Tensor] = None,
                 shape: Union[None, int, List[int], Tuple[int]] = None,
                 oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                 width: Union[int, List[int], Tuple[int]] = 3,
                 basis: Union[None, Tensor] = None,
                 device: str = 'cpu',
                 interpolator: Union[None, Dict] = None):

        super().__init__()

        # if provided, re-use precomputed interpolator.
        if interpolator is None:
            self.interpolator = _fourier.prepare_nufft(
                coord, shape, oversamp, width, basis, device)
        else:
            self.interpolator = interpolator

    @property
    def H(self):  # pylint: disable=invalid-name
        """Return adjoint linear operator. """


class _NUFFT(torch.autograd.Function):  # pylint: disable=abstract-method
    """ Autograd.Function to be used inside nn.Module. """
    @staticmethod
    def forward(ctx, image, interpolator):  # pylint: disable=abstract-method, arguments-differ
        return _fourier.nufft(image, interpolator)

    @staticmethod
    def backward(ctx, grad_kdata, interpolator):  # pylint: disable=arguments-differ
        return _fourier.nufft_adjoint(grad_kdata, interpolator)


class NUFFT(AbstractNUFFT):
    """ Non-Uniform Fourier Transform operator with embedded low-rank projection."""

    def forward(self, image):  # pylint: disable=missing-function-docstring
        return _NUFFT.apply(image, self.interpolator)

    @property
    def H(self):
        """ Return NUFFTAdjoint object. """
        return NUFFTAdjoint(interpolator=self.interpolator)


class _NUFFTAdjoint(torch.autograd.Function):  # pylint: disable=abstract-method
    """ Autograd.Function to be used inside nn.Module. """
    @staticmethod
    def forward(ctx, image, interpolator):  # pylint: disable=arguments-differ
        return _fourier.nufft_adjoint(image, interpolator)

    @staticmethod
    def backward(ctx, grad_kdata, interpolator):  # pylint: disable=arguments-differ
        return _fourier.nufft(grad_kdata, interpolator)


class NUFFTAdjoint(AbstractNUFFT):
    """ Adjoint Non-Uniform Fourier Transform operator with embedded low-rank projection."""

    def forward(self, kdata):  # pylint: disable=missing-function-docstring
        return _NUFFTAdjoint.apply(kdata, self.interpolator)

    @property
    def H(self):
        """ Return NUFFT object. """
        return NUFFT(interpolator=self.interpolator)


class _NUFFTSelfadjoint(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, image, toeplitz):  # pylint: disable=arguments-differ
        return _fourier.nufft_selfadjoint(image, toeplitz)

    @staticmethod
    def backward(ctx, grad_image, toeplitz):  # pylint: disable=arguments-differ
        return _fourier.nufft_selfadjoint(grad_image, toeplitz)


class NUFFTSelfadjoint(torch.nn.Module):
    """ Self-adjoint Non-Uniform Fourier Transform operator."""

    def __init__(self,
                 coord: Union[None, Tensor] = None,
                 shape: Union[None, int, List[int], Tuple[int]] = None,
                 oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                 width: Union[int, List[int], Tuple[int]] = 3,
                 device: str = 'cpu',
                 basis: Tensor = None,
                 dcf: Tensor = None,
                 toeplitz: Union[None, Dict] = None):
        super().__init__()

        # if provided, re-use precomputed interpolator.
        if toeplitz is None:
            self.toeplitz = _fourier.prepare_toeplitz(
                coord, shape, oversamp, width, basis, device, dcf)
        else:
            self.toeplitz = toeplitz

    @property
    def H(self):  # pylint: disable=invalid-name
        """Return adjoint linear operator. """
        return self

    def forward(self, image):  # pylint: disable=missing-function-docstring
        return _NUFFTSelfadjoint.apply(image, self.toeplitz)


