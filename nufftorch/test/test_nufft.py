# -*- coding: utf-8 -*-
"""Unit tests for forward NUFFT."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

import pytest


from torch import testing as tt


from torchmrifourier import functional


from conftest import _get_noncartesian_params


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(data_in, coord=coord, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave, basis", _get_noncartesian_params(lowrank=True))
def test_nufft_lowrank(ndim, device, img, kdata, wave, basis, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # computation
    result = functional.nufft(data_in, coord=coord, device=device, basis=basis)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_even_width_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # kernel width
    width = 4

    # computation
    result = functional.nufft(data_in, coord=coord, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_even_width_explicit_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # kernel width
    width = tuple([4] * ndim)

    # computation
    result = functional.nufft(data_in, coord=coord, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_odd_width_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # kernel width
    width = 3

    # computation
    result = functional.nufft(data_in, coord=coord, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_odd_width_explicit_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # kernel width
    width = tuple([3] * ndim)

    # computation
    result = functional.nufft(data_in, coord=coord, width=width, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)


@pytest.mark.parametrize("ndim, device, img, kdata, wave", _get_noncartesian_params())
def test_osf_nufft(ndim, device, img, kdata, wave, testing_tol, utils):

    # get input and output
    data_in = img
    expected = kdata

    # k-space coordinates
    coord = wave.coordinates

    # gridding oversampling factor
    oversamp = 1.125

    # computation
    result = functional.nufft(data_in, coord=coord,
                              oversamp=oversamp, device=device)
    result = utils.normalize(result)

    tt.assert_close(result, expected, rtol=testing_tol, atol=testing_tol)
