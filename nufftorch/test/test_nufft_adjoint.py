# -*- coding: utf-8 -*-
"""Unit tests for adjoint NUFFT."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

import pytest


from torch import testing as tt


from torchmrifourier import functional


from conftest import _get_noncartesian_params


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave, basis", _get_noncartesian_params(lowrank=True))
def test_nufft_adjoint_lowrank(ndim, device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave, share_width", _get_noncartesian_params(viewshare=True))
def test_nufft_adjoint_viewshare(ndim, device, img, kdata, wave, share_width, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, device=device, share_width=share_width)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave, basis, share_width", _get_noncartesian_params(lowrank=True, viewshare=True))
def test_nufft_adjoint_viewshare_lowrank(ndim, device, img, kdata, wave, basis, share_width, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, device=device, basis=basis, share_width=share_width)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_even_width_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # kernel width
    width = 4

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_even_width_explicit_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # kernel width
    width = tuple([4] * ndim)

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_odd_width_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # kernel width
    width = 3

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_odd_width_explicit_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # kernel width
    width = tuple([3] * ndim)

    # computation
    result = functional.nufft_adjoint(
        dcf * data_in, coord=coord, shape=shape, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_osf_nufft_adjoint(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = kdata
    expected = img

    # k-space coordinates
    coord = wave.coordinates
    dcf = wave.density_comp_factor
    shape = wave.acquisition_matrix

    # gridding oversampling factor
    oversamp = 1.125

    # computation
    result = functional.nufft_adjoint(dcf * data_in, coord=coord, shape=shape,
                                      oversamp=oversamp, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)
