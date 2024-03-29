# -*- coding: utf-8 -*-
"""
Basic test for test functionality.
"""
import numpy as np
import torch


from nufftorch import nn


import utils


import warnings

warnings.simplefilter('ignore', category=UserWarning)


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


def _adjoint(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    
    # setup
    ground_truth, basis, ktraj, dcf, shape = _setup_problem(npix, ndim, nechoes, nreadouts)
    
    # generate k-space data
    tkbF = utils.tkbnufft_factory(ktraj, im_size=shape)
    kdata = tkbF(ground_truth.clone().to(torch.complex64)).squeeze()
    kdata = kdata.reshape((kdata.shape[0], *ktraj.shape[:-1]))  
    kdata = (basis[:,:,None,None] * kdata).sum(axis=0)
        
    # reconstruct image using lr_nufft_torch
    FH = nn.NUFFTAdjoint(ktraj, shape=shape, basis=basis)
    image = FH(dcf * kdata.clone()).unsqueeze(axis=1)
    
    # reconstruct using torchkbnufft
    tkbFH = utils.tkbnufft_adjoint_factory(ktraj, im_size=shape)
    kdata = basis[:,:,None,None] * (dcf * kdata[None,...])
    kdata = kdata.reshape((kdata.shape[0], 1, np.prod(kdata.shape[1:])))  
    image_tkb = tkbFH(kdata.clone())
    
    return image, image_tkb, ground_truth
    

image, image_tkb, ground_truth = _adjoint(nechoes=1000, nreadouts=5000)
utils.show_image_series([torch.flip(ground_truth, dims=[-2,-1]), 
                         torch.abs(torch.flip(image.permute(0, 1, 3, 2), dims=[-2, -1])), 
                         torch.flip(image_tkb, dims=[-2,-1])], 
                        0, 
                        ylabel="torch-kb-nufft            nufft-torch             ground truth",
                        xlabel="$\phi_1$                  $\phi_2$                  $\phi_3$                  $\phi_4$")
