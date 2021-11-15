# -*- coding: utf-8 -*-
"""
Common routines for both CPU- and CUDA-based interpolation/gridding.

@author: Matteo Cencini
"""
from typing import Tuple

import itertools


class _iterator:  # pylint: disable=too-few-public-methods

    @staticmethod
    def _get_noncart_points_parallelize_over_all(index: int,
                                                 batch_size: int,
                                                 readout_length: int) -> Tuple[int, int, int]:

        frame = index // (batch_size*readout_length)
        tmp = index % (batch_size*readout_length)
        batch = tmp // readout_length
        point = tmp % readout_length

        return frame, batch, point

    @staticmethod
    def _nested_range(*args):
        return [range(i) for i in args]

    @staticmethod
    def _get_neighbourhood(*args):
        return tuple(itertools.product(_iterator._nested_range(*args)))

    @staticmethod
    def _check_boundaries(frame, nframes):

        frame = max(frame, 0)
        frame = min(frame, nframes)

        return frame


class _kernel:  # pylint: disable=too-few-public-methods

    @staticmethod
    def _prod(value_tuple: Tuple[float], row_index: int, col_index: int) -> float:

        value = value_tuple[0][row_index][col_index]

        for i in range(1, len(value_tuple)):
            value *= value_tuple[i][row_index][col_index]

        return value

    @staticmethod
    def _ravel_index(index_tuple: Tuple[float], grid_shape: Tuple[int],
                     row_index: int, col_index: int) -> int:

        index = index_tuple[0][row_index][col_index]

        for i in range(1, len(index_tuple)):
            index += index_tuple[i][row_index][col_index] * grid_shape[i-1]

        return index

    @staticmethod
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


class _gather:  # pylint: disable=too-few-public-methods

    @staticmethod
    def _data(data_out, data_in, index_tuple, kernel_value):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # get output and input locations
        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)

        # update data
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    def _data_lowrank(data_out, data_in, index_tuple, kernel_value, basis_adjoint):

        # unpack indexes
        frame, batch, index_out, index_in = index_tuple

        # get output locations
        idx_out = (frame, batch, index_out)

        # iterate over subspace coefficients
        for coeff in range(basis_adjoint.shape[-1]):
            # get input locations
            idx_in = (frame, batch, index_in)

            # get total weight
            weight = kernel_value * basis_adjoint[frame, coeff]

            # update data
            data_out[idx_out] += weight * data_in[idx_in]
