# -*- coding: utf-8 -*-
"""Unit tests for object-oriented versions of NUFFT, NUFFT adjoint and NUFFT self-adjoint routines, including autograd step."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

import pytest

import torch

from nufftorch import nn

from conftest import _get_noncartesian_params

# @pytest.mark.parametrize("ndim, device, img, wave, kdata", _get_noncartesian_params())
# def test_nufft(ndim, device, img, kdata, wave, utils):
    
#     # get ground truth
#     img_ground_truth = img.clone()
#     kdata_ground_truth = kdata.clone()

#     # k-space coordinates
#     coord = wave.coordinates
#     dcf = wave.density_comp_factor.to(kdata.device)
#     shape = wave.acquisition_matrix
    
#     # prepare operators
#     F = nn.NUFFT(coord=coord, shape=shape, device=device)
#     FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device)
      
#     # computation
#     kdata_out = F(img)
#     img_out = FH(dcf * kdata)
    
#     # check
#     a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
#     b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

#     assert torch.allclose(a, b)
 
    
# @pytest.mark.parametrize("ndim, device, img, wave, kdata, basis", _get_noncartesian_params(lowrank=True))
# def test_nufft_lowrank(ndim, device, img, kdata, wave, basis, utils):
    
#     # get ground truth
#     img_ground_truth = img.clone()
#     kdata_ground_truth = kdata.clone()

#     # k-space coordinates
#     coord = wave.coordinates
#     dcf = wave.density_comp_factor.to(kdata.device)
#     shape = wave.acquisition_matrix
    
#     # prepare operators
#     F = nn.NUFFT(coord=coord, shape=shape, device=device, basis=basis)
#     FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, basis=basis)
      
#     # computation
#     kdata_out = F(img)
#     img_out = FH(dcf * kdata)
    
#     # check
#     a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
#     b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

#     assert torch.allclose(a, b)
    
@pytest.mark.parametrize("ndim, device, img, wave", _get_noncartesian_params(selfadjoint=True))
def test_nufft_selfadjoint(ndim, device, img, wave, utils):
    
    # define tolerance
    tol = 1e-2  # toeplitz is only approximate
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(img.device)
    shape = wave.acquisition_matrix
    
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device)    
    FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf)
      
    # computation
    img_ground_truth = FH(dcf * F(img.clone()))
    img_toeplitz = FHF(img.clone())
    
    print(img_ground_truth[0,0,0,0])
    print(img_toeplitz[0,0,0,0])

    # check
    res = torch.norm(img_ground_truth - img_toeplitz) / torch.norm(img_ground_truth)

    assert res < tol
 
    
@pytest.mark.parametrize("ndim, device, img, wave, basis", _get_noncartesian_params(lowrank=True, selfadjoint=True))
def test_nufft_selfadjoint_lowrank(ndim, device, img, wave, basis, utils):
    
    # define tolerance
    tol = 1e-2  # toeplitz is only approximate
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(img.device)
    shape = wave.acquisition_matrix
    
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, basis=basis)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, basis=basis)    
    FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf, basis=basis)
      
    # computation
    img_ground_truth = FH(dcf * F(img.clone()))
    img_toeplitz = FHF(img.clone())
    
    # check
    res = torch.norm(img_ground_truth - img_toeplitz) / torch.norm(img_ground_truth)

    assert res < tol

