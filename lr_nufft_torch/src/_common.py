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


class _gather:  # pylint: disable=too-few-public-methods

    @staticmethod
    def _data(data_tuple, index_tuple, kernel_value):

        data_out, data_in = data_tuple
        frame, batch, index_out, index_in = index_tuple

        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    def _data_lowrank(data_tuple, index_tuple, kernel_value, basis_adjoint):

        data_out, data_in = data_tuple
        frame, batch, index_out, index_in = index_tuple

        idx_out = (frame, batch, index_out)

        for coeff in range(basis_adjoint.shape[-1]):
            idx_in = (frame, batch, index_in)
            weight = kernel_value * basis_adjoint[frame, coeff]

            data_out[idx_out] += weight * data_in[idx_in]
