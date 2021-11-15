# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:43:40 2020

@author: mcencini
"""
import pytest

from torch import testing as tt

from lr_nufft_torch import functional

from conftest import (get_params_2d_nufft,
                      get_params_3d_nufft,
                      get_params_2d_nufft_lowrank,
                      get_params_3d_nufft_lowrank)

# %% testing kernel width


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_2d_nufft(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_2d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=4, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_even_width_explicit_2d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=(4, 4), device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_2d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=5, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_odd_width_explicit_2d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=(5, 5), device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=4, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_even_width_explicit_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(4, 4, 4), device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, width=5, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_odd_width_explicit_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(5, 5, 5), device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor

@pytest.mark.parametrize("device, img, kdata, wave", get_params_2d_nufft())
def test_osf_2d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, oversamp=2, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave", get_params_3d_nufft())
def test_osf_3d_nufft(device, img, kdata, wave, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, oversamp=2, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% low rank

@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(input, coord=coord, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_even_width_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=4, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_even_width_explicit_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(4, 4), device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_odd_width_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=5, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_odd_width_explicit_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(5, 5), device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_even_width_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=4, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_even_width_explicit_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(4, 4, 4), device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_odd_width_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input,  coord=coord, width=5, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_odd_width_explicit_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, width=(5, 5, 5), device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


# %% testing oversampling factor: precomputed nufft

@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_2d_nufft_lowrank())
def test_osf_2d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, oversamp=2, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("device, img, kdata, wave, basis", get_params_3d_nufft_lowrank())
def test_osf_3d_nufft_lowrank(device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    input = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(
        input, coord=coord, oversamp=2, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)
