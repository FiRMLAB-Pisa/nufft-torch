# -*- coding: utf-8 -*-
"""
Basic test for test functionality.
"""
from lr_nufft_torch import _test_utils as utils
from lr_nufft_torch import module

from lr_nufft_torch._util import prod

import torch

import numpy as np

def _setup_problem(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    if ndim == 2:
        shape = (npix, npix)
    else:
        shape = (npix, npix, npix)
    
    # generate data
    ground_truth = utils.create_shepp_logan_phantom(shape)
    
    # generate trajectory
    ktraj, dcf = utils.create_radial_trajectory(ndim, npix, nreadouts, nechoes)
    
    # generate low rank basis
    basis, _ = utils.create_low_rank_subspace_basis(nechoes=nechoes)
    
    return ground_truth, basis, ktraj, dcf, shape

def test_adjoint(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    
    # setup
    ground_truth, basis, ktraj, dcf, shape = _setup_problem(npix, ndim, nechoes, nreadouts)
    
    # generate k-space data
    tkbF = utils.tkbnufft_factory(ktraj, im_size=shape)
    kdata = tkbF(ground_truth.clone().to(torch.complex64)).squeeze()
    kdata = kdata.reshape((kdata.shape[0], *ktraj.shape[:-1]))  
    kdata = (basis[:,:,None,None] * kdata).sum(axis=0)
        
    # reconstruct image using lr_nufft_torch
    FH = module.NUFFTAdjoint(ktraj, shape=shape, basis=basis)
    image = FH(dcf * kdata.clone()).unsqueeze(axis=1)
    
    # reconstruct using torchkbnufft
    tkbFH = utils.tkbnufft_adjoint_factory(ktraj, im_size=shape)
    kdata = basis[:,:,None,None] * (dcf * kdata[None,...])
    kdata = kdata.reshape((kdata.shape[0], 1, prod(kdata.shape[1:])))  
    image_tkb = tkbFH(kdata.clone())
    
    return image, image_tkb, ground_truth
    
def test_selfadjoint(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    
    # setup
    ground_truth, basis, ktraj, dcf, shape = _setup_problem(npix, ndim, nechoes, nreadouts)
    
    # generate k-space data
    tkbF = utils.tkbnufft_factory(ktraj, im_size=shape)
    kdata = tkbF(ground_truth.clone().to(torch.complex64)).squeeze()
    kdata = kdata.reshape((kdata.shape[0], *ktraj.shape[:-1]))  
    kdata = (basis[:,:,None,None] * kdata).sum(axis=0)
        
    # reconstruct image using lr_nufft_torch
    FH = module.NUFFTAdjoint(ktraj, shape=shape, basis=basis)
    image = FH(dcf * kdata.clone()).unsqueeze(axis=1)
    
    # prepare self-adjoint
    FHF = module.NUFFTSelfadjoint(ktraj, shape=shape, basis=basis, dcf=dcf)
    image = FHF(image.clone())
    
    # reconstruct using torchkbnufft
    tkbFH = utils.tkbnufft_adjoint_factory(ktraj, im_size=shape)
    kdata = basis[:,:,None,None] * (dcf * kdata[None,...])
    kdata = kdata.reshape((kdata.shape[0], 1, prod(kdata.shape[1:])))  
    image_tkb = tkbFH(kdata.clone())
    
    # prepare self-adjoint
    tkbFHF = utils.tkbnufft_selfadjoint_factory(ktraj, im_size=shape, dcf=dcf)
    image_tkb = tkbFHF(image_tkb.clone())
    
    return image, image_tkb, ground_truth

image, image_tkb, ground_truth = test_adjoint()
utils.show_image_series([torch.flip(ground_truth, dims=[-2,-1]), torch.abs(image), torch.flip(image_tkb, dims=[-2,-1])], 0)
