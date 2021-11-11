#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:43:40 2020

@author: mcencini
"""
import pytest

import sigpy

from torch import testing as tt

import lr_nufft_torch.fourier as fourier

from .conftest import (get_params_2d_nufft,
                       get_params_3d_nufft,
                       get_params_2d_nufft_lowrank,
                       get_params_3d_nufft_lowrank)

# %% testing kernel width: precomputed nufft


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=4)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_explicit_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=(4, 4))

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=5)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_explicit_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=(5, 5))

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=4)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_explicit_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=(4, 4, 4))

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=5)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_explicit_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, width=(5, 5, 5))

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor: precomputed nufft

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_osf_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, oversamp=2)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_osf_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape, oversamp=2)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing non-scalar input for image size
@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_non_scalar_shape_2d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape)

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_non_scalar_shape_3d_nufft_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape, shape)

    # build nufft object
    nufft_ob = fourier.prepare_nufft(coord, shape)

    # computation
    result = fourier.nufft(input, nufft_ob)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)
