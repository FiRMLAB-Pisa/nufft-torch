# -*- coding: utf-8 -*-
"""
Autograd wrappers for Fourier routines.

@author: Matteo Cencini
"""
# pylint: disable=abstract-method
# pylint: disable=arguments-differ

import torch


from nufftorch.src import _routines


class _nufft(torch.autograd.Function):
    """ Autograd.Function to be used inside nn.Module. """
    @staticmethod
    def forward(ctx, image, interpolator):  
        return _routines.nufft(image, interpolator)

    @staticmethod
    def backward(ctx, grad_kdata, interpolator):
        return _routines.nufft_adjoint(grad_kdata, interpolator)
   
    
class _nufft_adjoint(torch.autograd.Function):
    """ Autograd.Function to be used inside nn.Module. """
    @staticmethod
    def forward(ctx, image, interpolator):
        return _routines.nufft_adjoint(image, interpolator)

    @staticmethod
    def backward(ctx, grad_kdata, interpolator):
        return _routines.nufft(grad_kdata, interpolator)
    
    
class _nufft_selfadjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, interpolator):
        return _routines.toeplitz_convolution(image, interpolator)

    @staticmethod
    def backward(ctx, grad_image, interpolator):
        return _routines.toeplitz_convolution(grad_image, interpolator)

