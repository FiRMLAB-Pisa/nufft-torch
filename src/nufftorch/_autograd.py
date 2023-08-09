"""Autograd wrappers for Fourier routines."""

import torch

from nufftorch.subroutines import _routines


class _nufft(torch.autograd.Function):
    """Autograd.Function to be used inside nn.Module."""

    @staticmethod
    def forward(ctx, image, interpolator):
        ctx.interpolator = interpolator
        return _routines.nufft(image, interpolator)

    @staticmethod
    def backward(ctx, grad_kdata):
        interpolator = ctx.interpolator
        return _routines.nufft_adjoint(grad_kdata, interpolator), None


class _nufft_adjoint(torch.autograd.Function):
    """Autograd.Function to be used inside nn.Module."""

    @staticmethod
    def forward(ctx, kdata, interpolator):
        ctx.interpolator = interpolator
        return _routines.nufft_adjoint(kdata, interpolator)

    @staticmethod
    def backward(ctx, grad_image):
        interpolator = ctx.interpolator
        return _routines.nufft(grad_image, interpolator), None


class _nufft_selfadjoint(torch.autograd.Function):
    """Autograd.Function to be used inside nn.Module."""

    @staticmethod
    def forward(ctx, image, interpolator):
        ctx.interpolator = interpolator
        return _routines.toeplitz_convolution(image, interpolator)

    @staticmethod
    def backward(ctx, grad_image):
        interpolator = ctx.interpolator
        return _routines.toeplitz_convolution(grad_image, interpolator), None
