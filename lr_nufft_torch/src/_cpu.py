# -*- coding: utf-8 -*-
"""
CPU specific functions.

@author: Matteo
"""
from typing import Tuple

import numpy as np
import numba as nb

from lr_nufft_torch.src import _common, _util


# %% Utils

# def _interpolate():

#     # get sizes
#     nframes, batch_size, _, _ = cart_data.shape
#     npts = noncart_data.shape[-1]

#     # parallelize over frames, batches and k-space points
#     for i in nb.prange(n_target_points):  # pylint: disable=not-an-iterable

#         # get current frame and k-space index
#         target_point =
#         # gather data within kernel radius
#         for i_y in range(ywidth):
#             idy = yindex[frame, point, i_y]
#             valy = yvalue[frame, point, i_y]

#             for i_x in range(xwidth):
#                 idx = xindex[frame, point, i_x]
#                 val = valy * xvalue[frame, point, i_x]

#                 noncart_data[frame, batch, point] += \
#                     val * cart_data[frame, batch, idy, idx]

class _iterator(_common._iterator):  # pylint: disable=too-few-public-methods

    _get_noncart_points_parallelize_over_all = staticmethod(nb.njit(
        _common._iterator._get_noncart_points_parallelize_over_all, fastmath=True, cache=True))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_noncart_points_parallelize_over_batch_and_frame(index: int, batch_size: int) -> Tuple[int, int]:

        frame = index // batch_size
        batch = index % batch_size

        return frame, batch


class _kernel(_common._kernel):  # pylint: disable=too-few-public-methods

    @staticmethod
    @nb.njit(fastmath=True, cache=True)  # pragma: no cover
    def _function(x_0, beta):
        if abs(x_0) > 1:
            value = 0

        x_0 = beta * (1 - x_0**2)**0.5
        x_i = x_0 / 3.75
        if x_i < 3.75:
            value = 1 + 3.5156229 * x_i**2 + 3.0899424 * x_i**4 +\
                1.2067492 * x_i**6 + 0.2659732 * x_i**8 +\
                0.0360768 * x_i**10 + 0.0045813 * x_i**12
        else:
            value = x_0**-0.5 * np.exp(x_0) * (
                0.39894228 + 0.01328592 * x_i**-1 +
                0.00225319 * x_i**-2 - 0.00157565 * x_i**-3 +
                0.00916281 * x_i**-4 - 0.02057706 * x_i**-5 +
                0.02635537 * x_i**-6 - 0.01647633 * x_i**-7 +
                0.00392377 * x_i**-8)
            
        return value

    _prod = staticmethod(
        nb.njit(_common._kernel._prod, fastmath=True, cache=True))
    _ravel_index = staticmethod(
        nb.njit(_common._kernel._ravel_index, fastmath=True, cache=True))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _evaluate(sample_index: int,
                  neighbour_index: int,
                  kernel_tuple: Tuple[Tuple[float],
                                      Tuple[int],
                                      Tuple[int]]) -> Tuple[float, int]:

        # unpack kernel tuple
        value_tuple, index_tuple, grid_shape = kernel_tuple

        # get kernel value
        value = _kernel._prod(
            value_tuple, sample_index, neighbour_index)

        # get flattened neighbour index
        index = _kernel._ravel_index(
            index_tuple, grid_shape, sample_index, neighbour_index)

        return value, index


class _gather(_common._gather):  # pylint: disable=too-few-public-methods

    _data = staticmethod(
        nb.njit(_common._gather._data, fastmath=True, cache=True))
    _ravel_index = staticmethod(
        nb.njit(_common._gather._data_lowrank, fastmath=True, cache=True))


class _spread:  # pylint: disable=too-few-public-methods

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data(data_tuple, index_tuple, kernel_value):

        data_out, data_in = data_tuple
        frame, batch, index_out, index_in = index_tuple

        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data_lowrank(data_tuple, index_tuple, kernel_value, basis):

        data_out, data_in = data_tuple
        coeff, batch, index_out, index_in = index_tuple

        idx_out = (coeff, batch, index_out)

        for frame in range(basis.shape[-1]):
            idx_in = (frame, batch, index_in)
            weight = kernel_value * basis[coeff, frame]

            data_out[idx_out] += weight * data_in[idx_in]
