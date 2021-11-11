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


# %% testing kernel width: precomputed t_nufft_adjoint
@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())
def test_even_width_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix

    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(6, 6))

    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())
def test_odd_width_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix

    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(7, 7))

    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())
def test_even_width_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix

    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(6, 6, 6))

    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_t_nufft())
def test_odd_width_3d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix

    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape, width=(7, 7, 7))

    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor: precomputed t_nufft_adjoint


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())
def test_osf_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

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


# %% testing non-scalar input for image size: precomputed nufft_adjoint

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_t_nufft())
def test_non_scalar_shape_2d_t_nufft_adjoint_precomp(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = kdata
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape, shape)

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
    expected = sigpy.to_device(img, -1)  # bring expected result to CPU

    # k-space coordinates
    coord = wave.coordinates
    dcf = fourier.tDensityCompensationFactor(wave.density_comp_factor)
    shape = wave.acquisition_matrix
    shape = (shape, shape, shape)

    # build t_nufft object
    t_nufft_ob = fourier.prepare_t_nufft(coord, shape)

    # computation
    result = fourier.t_nufft_adjoint(dcf.apply(input), t_nufft_ob)
    result = sigpy.to_device(result, -1)
    result = utils.normalize(result)

    npt.assert_allclose(result, expected, rtol=testing_tol, atol=testing_tol)
