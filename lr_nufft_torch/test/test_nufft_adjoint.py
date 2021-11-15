# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:43:40 2020

@author: mcencini
"""
import pytest

from torch import testing as tt

from lr_nufft_torch import  functional

from conftest import (get_params_2d_nufft,
                      get_params_3d_nufft,
                      get_params_2d_nufft_lowrank,
                      get_params_3d_nufft_lowrank)

# %% testing kernel width


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord, shape, width=4)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_explicit_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=(4, 4))
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=5)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_explicit_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=(5, 5))
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=4)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_explicit_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=(4, 4, 4))
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=5)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_explicit_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, width=(5, 5, 5))
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_osf_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, oversamp=2)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_osf_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, oversamp=2)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing non-scalar input for image size
@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_non_scalar_shape_2d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape)

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_non_scalar_shape_3d_nufft_adjoint(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape, shape)

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)

# %% low rank


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_even_width_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, cood=coord, shape=shape, width=4, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_even_width_explicit_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=(4, 4), basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_odd_width_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=5, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_odd_width_explicit_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=(5, 5), basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_even_width_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=4, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_even_width_explicit_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=(4, 4, 4), basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_odd_width_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input,  coord=coord, shape=shape, width=5, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_odd_width_explicit_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, width=(5, 5, 5), basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor

@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_osf_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, oversamp=2, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_osf_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        input, coord=coord, shape=shape, oversamp=2, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing non-scalar input for image size
@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_non_scalar_shape_2d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape)

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_non_scalar_shape_3d_nufft_adjoint_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    shape = (shape, shape, shape)

    # computation
    result = functional.nufft_adjoint(input, coord=coord, shape=shape, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)
