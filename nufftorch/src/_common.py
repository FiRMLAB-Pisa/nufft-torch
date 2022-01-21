# -*- coding: utf-8 -*-
"""
Common subroutines for CPU- and CUDA-based interpolation/gridding.

@author: Matteo Cencini
"""
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

from typing import Tuple

import itertools


class _iterator:

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
        return [list(range(i)) for i in args]

    @staticmethod
    def _get_neighbourhood(*args):
        return tuple(itertools.product(*_iterator._nested_range(*args)))


class _kernel:

    @staticmethod
    def _prod(value_arr, row_index, col_index, slice_index):
        value = value_arr[0][row_index][col_index][slice_index[0]]
        
        for i in range(1, value_arr.shape[0]):
            value *= value_arr[i][row_index][col_index][slice_index[i]]

        return value

    @staticmethod
    def _ravel_index(index_arr, grid_offset, row_index, col_index, slice_index):
        index = index_arr[0][row_index][col_index][slice_index[0]]

        for i in range(1, index_arr.shape[0]):
            index += index_arr[i][row_index][col_index][slice_index[i]] * grid_offset[i-1]

        return index

    @staticmethod
    def _make_evaluate(_prod, _ravel_index):   
        def _evaluate(frame, sample_idx, neighbour_idx, kernel_value, kernel_idx, kernel_width, grid_offset):  
            # get kernel value
            value = _prod(kernel_value, frame, sample_idx, neighbour_idx)
    
            # get flattened neighbour index
            index = _ravel_index(kernel_idx, grid_offset, frame, sample_idx, neighbour_idx)
            
            return value, index
        
        return _evaluate


class _gather:

    @staticmethod
    def _data(data_out, data_in, frame, batch, index_out, index_in, kernel_value):
        # get output and input locations
        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)

        # update data
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    def _data_lowrank(data_out, data_in, frame, batch, index_out, index_in, kernel_value, basis_adjoint, ncoeff):
        # get output locations
        idx_out = (frame, batch, index_out)
        
        # iterate over subspace coefficients
        for coeff in range(ncoeff):
            # get input locations
            idx_in = (frame, batch, index_in)

            # get total weight
            weight = kernel_value * basis_adjoint[frame, coeff]

            # update data
            data_out[idx_out] += weight * data_in[idx_in]


def _dot_product(out, in_a, in_b):
    row, col = in_a.shape

    for j in range(col):
        for i in range(row):
            out[j] += in_a[i][j] * in_b[j]