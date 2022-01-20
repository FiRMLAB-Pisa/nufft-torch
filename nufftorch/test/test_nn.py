# -*- coding: utf-8 -*-
"""Unit tests for object-oriented versions of NUFFT, NUFFT adjoint and NUFFT self-adjoint routines, including autograd step."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument


import itertools
import pytest


import torch


from nufftorch import nn


from conftest import _kt_space_data, _kt_space_trajectory, _image, _lowrank_subspace_projection, _utils


# test values
ndim = [3]
device = ['cpu']
dtype = [torch.float32, torch.complex64]

nframes = [1, 2]
nechoes = [1, 2]
ncoils = [1, 2]
nslices = [1, 2]

# if torch.cuda.is_available():
#     device += ['cuda']


# @pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
#     list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
# def test_nufft(ndim, nframes, nechoes, ncoils, nslices, device, 
#                npix=8, osf=1.125, width=3, test=True):
        
#     # get ground truth
#     img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
#     kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

#     # k-space coordinates
#     wave = _kt_space_trajectory(ndim, nframes, npix)
#     coord = wave.coordinates
#     dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
#     shape = wave.acquisition_matrix
      
#     # prepare operators
#     F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
#     FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
      
#     # computation
#     kdata_out = F(img_ground_truth.clone())
#     img_out = FH(dcf * kdata_ground_truth.clone())
    
#     # check
#     a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
#     b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

#     if test:
#       assert torch.allclose(a, b)
    
#     return img_out.squeeze().detach(), kdata_out.squeeze().detach()
 
    
# @pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
#     list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
# def test_nufft_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, npix=8, test=True):
        
#     # get ground truth
#     img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
#     kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

#     # k-space coordinates
#     wave = _kt_space_trajectory(ndim, nframes, npix)
#     coord = wave.coordinates
#     dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
#     shape = wave.acquisition_matrix
    
#     # get basis
#     basis = _lowrank_subspace_projection(dtype, nframes)
      
#     # prepare operators
#     F = nn.NUFFT(coord=coord, shape=shape, device=device, basis=basis)
#     FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, basis=basis)
      
#     # computation
#     kdata_out = F(img_ground_truth.clone())
#     img_out = FH(dcf * kdata_ground_truth.clone())
    
#     # check
#     a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
#     b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

#     if test:
#       assert torch.allclose(a, b)
    
#     return img_out.squeeze().detach(), kdata_out.squeeze().detach()

    
# @pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
#     list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
# def test_nufft_selfadjoint(ndim, nframes, nechoes, ncoils, nslices, device, 
#                            npix=8, osf=1.125, width=3, test=True):
    
#     # define tolerance
#     tol = 1e-1  # toeplitz is only approximate
    
#     # get ground truth
#     img = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
#     # k-space coordinates
#     wave = _kt_space_trajectory(ndim, nframes, npix)
#     coord = wave.coordinates
#     dcf = wave.density_comp_factor.to(img.device)
#     shape = wave.acquisition_matrix
    
#     # prepare operators
#     F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
#     FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width)    
#     FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf, prep_oversamp=osf, width=width)
      
#     # computation
#     img_ground_truth = FH(dcf * F(img.clone()))
#     img_toeplitz = FHF(img.clone())
        
#     # check
#     res = torch.norm(img_ground_truth.abs() - img_toeplitz.abs()) / torch.norm(img_ground_truth.abs())

#     if test:
#         assert res < tol
        
#     return img_ground_truth.squeeze().detach(), img_toeplitz.squeeze().detach()


@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_selfadjoint_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                                   npix=8, osf=1.125, width=3, test=True):
    
    # define tolerance
    tol = 1e-1  # toeplitz is only approximate
    
    # get ground truth
    img = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(img.device)
    shape = wave.acquisition_matrix
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
    
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, basis=basis, oversamp=osf, width=width)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, basis=basis, oversamp=osf, width=width)    
    FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf, basis=basis, prep_oversamp=osf, width=width)
      
    # computation
    img_ground_truth = FH(dcf * F(img.clone()))
    img_toeplitz = FHF(img.clone())
        
    # check
    res = torch.norm(img_ground_truth.abs() - img_toeplitz.abs()) / torch.norm(img_ground_truth.abs())

    if test:
        assert res < tol
    
    return img_ground_truth.squeeze().detach(), img_toeplitz.squeeze().detach()
 
