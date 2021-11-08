#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:02:54 2021

@author: mcencini
"""
import pytest 

import sigpy

from numpy import testing as npt

import qti.fourier_transform.fourier as fourier


from .conftest import (get_params_1d_t_nufft, 
                       get_params_2d_t_nufft,
                       get_params_3d_t_nufft,
                       get_params_1d_t_nufft_lowrank, 
                       get_params_2d_t_nufft_lowrank,
                       get_params_3d_t_nufft_lowrank,
                       get_params_1d_t_nufft_viewshare, 
                       get_params_2d_t_nufft_viewshare,
                       get_params_3d_t_nufft_viewshare,
                       get_params_1d_t_nufft_viewshare_lowrank, 
                       get_params_2d_t_nufft_viewshare_lowrank,
                       get_params_3d_t_nufft_viewshare_lowrank)

    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())
  
def test_default_1d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_default_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_default_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_1d_t_nufft_lowrank())


def test_default_1d_t_nufft_adjoint_lowrank(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_2d_t_nufft_lowrank())

def test_default_2d_t_nufft_adjoint_lowrank(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_3d_t_nufft_lowrank())

def test_default_3d_t_nufft_adjoint_lowrank(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_1d_t_nufft_viewshare())


def test_default_1d_t_nufft_adjoint_viewshare(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_2d_t_nufft_viewshare())

def test_default_2d_t_nufft_adjoint_viewshare(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_3d_t_nufft_viewshare())

def test_default_3d_t_nufft_adjoint_viewshare(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_1d_t_nufft_viewshare_lowrank())


def test_default_1d_t_nufft_adjoint_viewshare_lowrank(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_2d_t_nufft_viewshare_lowrank())

def test_default_2d_t_nufft_adjoint_viewshare_lowrank(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_3d_t_nufft_viewshare_lowrank())

def test_default_3d_t_nufft_adjoint_viewshare_lowrank(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
#%% testing kernel width: on-the-fly t_nufft_adjoint    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_even_width_1d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=6)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_odd_width_1d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=7)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=6)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_explicit_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=(6,6))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=7)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_explicit_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=(7,7))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=6)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_explicit_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=(6,6,6))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=7)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_explicit_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, width=(7,7,7))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

#%% testing oversampling factor: on-the-fly t_nufft_adjoint    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_osf_1d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_osf_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_osf_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
#%% testing non-scalar input for image size: on-the-fly nufft_adjoint

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())


def test_non_scalar_shape_2d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape,shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())


def test_non_scalar_shape_3d_t_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape,shape,shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), coord, shape)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


#%% testing default behaviour: precomputed t_nufft_adjoint

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())

# checking result
def test_default_1d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

# checking result
def test_default_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

# checking result
def test_default_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_1d_t_nufft_lowrank())

# checking result
def test_default_1d_t_nufft_adjoint_lowrank_precomp(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_2d_t_nufft_lowrank())

# checking result
def test_default_2d_t_nufft_adjoint_lowrank_precomp(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, LR", get_params_3d_t_nufft_lowrank())

# checking result
def test_default_3d_t_nufft_adjoint_lowrank_precomp(device, img, kdata, wave, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_1d_t_nufft_viewshare())

# checking result
def test_default_1d_t_nufft_adjoint_viewshare_precomp(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_2d_t_nufft_viewshare())

# checking result
def test_default_2d_t_nufft_adjoint_viewshare_precomp(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object", get_params_3d_t_nufft_viewshare())

# checking result
def test_default_3d_t_nufft_adjoint_viewshare_precomp(device, img, kdata, wave, share_object, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_1d_t_nufft_viewshare_lowrank())

# checking result
def test_default_1d_t_nufft_adjoint_viewshare_lowrank_precomp(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_2d_t_nufft_viewshare_lowrank())

# checking result
def test_default_2d_t_nufft_adjoint_viewshare_lowrank_precomp(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, share_object, LR", get_params_3d_t_nufft_viewshare_lowrank())

# checking result
def test_default_3d_t_nufft_adjoint_viewshare_lowrank_precomp(device, img, kdata, wave, share_object, LR, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob, share_object=share_object, LR=LR)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


#%% testing kernel width: precomputed t_nufft_adjoint    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_even_width_1d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=6)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_odd_width_1d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=7)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft_adjoint object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=6)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_explicit_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(6,6))
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=7)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_explicit_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(7,7))
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=6)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_explicit_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(6,6,6))
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=7)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_explicit_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(7,7,7))
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

#%% testing oversampling factor: precomputed t_nufft_adjoint    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_osf_1d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(expected, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_osf_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_osf_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
   
#%% testing non-scalar input for image size: precomputed nufft_adjoint

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())


def test_non_scalar_shape_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape,shape)
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())


def test_non_scalar_shape_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = kdata
    expected = sigpy.to_device(img,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape,shape,shape)
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
