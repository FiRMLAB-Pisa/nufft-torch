#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:43:40 2020

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
                       get_params_3d_t_nufft_lowrank)

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_default_1d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_default_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_default_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_1d_t_nufft_lowrank())


def test_default_1d_t_nufft_lowrank(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_2d_t_nufft_lowrank())

def test_default_2d_t_nufft_lowrank(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_3d_t_nufft_lowrank())

def test_default_3d_t_nufft_lowrank(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
#%% testing kernel width: on-the-fly t_nufft    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_even_width_1d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=4)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_odd_width_1d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=5)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=4)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_explicit_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=(4,4))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=5)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_explicit_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=(5,5))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=4)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_explicit_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=(4,4,4))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=5)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_explicit_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, width=(5,5,5))
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

#%% testing oversampling factor: on-the-fly t_nufft    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_osf_1d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_osf_2d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_osf_3d_t_nufft(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    
    # computation
    result = fourier.t_nufft(input, coord, oversamp=2)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)

#%% testing default behaviour: precomputed t_nufft

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())

# checking result
def test_default_1d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

# checking result
def test_default_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

# checking result
def test_default_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_1d_t_nufft_lowrank())

# checking result
def test_default_1d_t_nufft_lowrank_precomp(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_2d_t_nufft_lowrank())

# checking result
def test_default_2d_t_nufft_lowrank_precomp(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave, LRH", get_params_3d_t_nufft_lowrank())

# checking result
def test_default_3d_t_nufft_lowrank_precomp(device, img, kdata, wave, LRH, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob, LRH=LRH)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


#%% testing kernel width: precomputed t_nufft    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_even_width_1d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=4)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_odd_width_1d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=5)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=4)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_even_width_explicit_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(4,4))
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=5)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_odd_width_explicit_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(5,5))
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
 
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=4)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_even_width_explicit_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(4,4,4))
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=5)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_odd_width_explicit_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(5,5,5))
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

#%% testing oversampling factor: precomputed t_nufft    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_1d_t_nufft())


def test_osf_1d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(expected, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())

def test_osf_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    

@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())

def test_osf_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, oversamp=2)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
   
#%% testing non-scalar input for image size

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())


def test_non_scalar_shape_2d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape,shape)
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
        
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
    
@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())


def test_non_scalar_shape_3d_t_nufft_precomp(device, img, kdata, wave, testing_tol, utils):
    
    # get input and output
    input = img
    expected = sigpy.to_device(kdata,-1) # bring expected result to CPU
    
    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape,shape,shape)
    
    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)
    
    # computation
    result = fourier.t_nufft(input, t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)
      
    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)

