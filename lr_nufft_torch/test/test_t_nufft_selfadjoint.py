#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:27:07 2021

@author: mcencini
"""
import pytest 
import numpy as np

import sigpy

from numpy import testing as npt

import qti.fourier_transform.fourier as fourier

from .conftest import (get_params_1d_t_nufft_selfadjoint, 
                       get_params_2d_t_nufft_selfadjoint,
                       get_params_3d_t_nufft_selfadjoint,
                       get_params_1d_t_nufft_selfadjoint_lowrank, 
                       get_params_2d_t_nufft_selfadjoint_lowrank,
                       get_params_3d_t_nufft_selfadjoint_lowrank,
                       get_params_1d_t_nufft_selfadjoint_viewshare, 
                       get_params_2d_t_nufft_selfadjoint_viewshare,
                       get_params_3d_t_nufft_selfadjoint_viewshare,
                       get_params_1d_t_nufft_selfadjoint_viewshare_lowrank, 
                       get_params_2d_t_nufft_selfadjoint_viewshare_lowrank,
                       get_params_3d_t_nufft_selfadjoint_viewshare_lowrank)


@pytest.mark.parametrize("device, img, wave", get_params_1d_t_nufft_selfadjoint())

def test_default_results_1d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf)

    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave", get_params_2d_t_nufft_selfadjoint())

def test_default_results_2d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, wave", get_params_2d_t_nufft_selfadjoint())

def test_explicit_shape_results_2d(device, img, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    shape = (shape, shape)
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, wave", get_params_3d_t_nufft_selfadjoint())

def test_default_results_3d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave", get_params_3d_t_nufft_selfadjoint())

def test_explicit_shape_results_3d(device, img, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    shape = (shape, shape, shape)
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
               
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave, LR", get_params_1d_t_nufft_selfadjoint_lowrank())

def test_default_results_1d_lowrank(device, img, wave, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, LR=LR)

    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave, LR", get_params_2d_t_nufft_selfadjoint_lowrank())

def test_default_results_2d_lowrank(device, img, wave, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, LR=LR)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)

@pytest.mark.parametrize("device, img, wave, LR", get_params_3d_t_nufft_selfadjoint_lowrank())

def test_default_results_3d_lowrank(device, img, wave, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, LR=LR)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, wave, share_object", get_params_1d_t_nufft_selfadjoint_viewshare())

def test_default_results_1d_viewshare(device, img, wave, share_object, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object)

    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave, share_object", get_params_2d_t_nufft_selfadjoint_viewshare())

def test_default_results_2d_viewshare(device, img, wave, share_object, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)

@pytest.mark.parametrize("device, img, wave, share_object", get_params_3d_t_nufft_selfadjoint_viewshare())

def test_default_results_3d_viewshare(device, img, wave, share_object, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)  
    
    
@pytest.mark.parametrize("device, img, wave, share_object, LR", get_params_1d_t_nufft_selfadjoint_viewshare_lowrank())

def test_default_results_1d_viewshare_lowrank(device, img, wave, share_object, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object, LR=LR)

    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, wave, share_object, LR", get_params_2d_t_nufft_selfadjoint_viewshare_lowrank())

def test_default_results_2d_viewshare_lowrank(device, img, wave, share_object, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object, LR=LR)

    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)

@pytest.mark.parametrize("device, img, wave, share_object, LR", get_params_3d_t_nufft_selfadjoint_viewshare_lowrank())

def test_default_results_3d_viewshare_lowrank(device, img, wave, share_object, LR, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, 
                                                   share_object=share_object, LR=LR)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
#%% testing behaviour when building toeplitz with precomputed nufft: result correctness
   
@pytest.mark.parametrize("device, img, wave", get_params_1d_t_nufft_selfadjoint())

def test_precomp_results_1d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, precomp=True)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, wave", get_params_2d_t_nufft_selfadjoint())

def test_precomp_results_2d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, precomp=True)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, wave", get_params_3d_t_nufft_selfadjoint())

def test_precomp_results_3d(device, img, wave, testing_tol, utils, caplog):
    
    # get input and output
    input = img
    expected = np.copy(sigpy.to_device(img, -1)) # bring expected result to CPU
    
    # k-space coordinates
    coord = sigpy.to_device(wave.coordinates, device)
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix
    
    # computation
    st_kernel = fourier.prepare_tNUFFT_selfadjoint(coord, shape, dcf=dcf, precomp=True)
    
    result = fourier.apply_t_selfadjoint(input, st_kernel)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
