# -*- coding: utf-8 -*-
"""Unit tests for adjoint NUFFT."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

import pytest


import torch


from nufftorch import functional


from conftest import _get_noncartesian_params

@pytest.mark.parametrize("ndim, device, img, wave, kdata", _get_noncartesian_params())
def test_nufft(ndim, device, img, kdata, wave, utils):
    
    # get ground truth
    img_ground_truth = img.clone()
    kdata_ground_truth = kdata.clone()

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
      
    # computation
    kdata_out = functional.nufft(img, coord=coord, device=device)
    img_out = functional.nufft_adjoint(dcf * kdata, coord=coord, shape=shape, device=device)
    
    # check
    a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
    b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

    assert torch.allclose(a, b)
 
    
@pytest.mark.parametrize("ndim, device, img, wave, kdata, basis", _get_noncartesian_params(lowrank=True))
def test_nufft_lowrank(ndim, device, img, kdata, wave, basis, utils):
    
    # get ground truth
    img_ground_truth = img.clone()
    kdata_ground_truth = kdata.clone()

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
      
    # computation
    kdata_out = functional.nufft(img, coord=coord, device=device, basis=basis)
    img_out = functional.nufft_adjoint(dcf * kdata, coord=coord, shape=shape, device=device, basis=basis)
    
    # check
    a = torch.inner(img_out.flatten(), img_ground_truth.flatten())
    b = torch.inner(kdata_out.flatten(), kdata_ground_truth.flatten())

    assert torch.allclose(a, b)
