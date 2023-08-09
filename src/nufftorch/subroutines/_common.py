"""Common subroutines for CPU- and CUDA-based interpolation/gridding."""

from typing import Tuple

import itertools


def _dot_product(out, in_a, in_b):
    """Naive implementation of matrix-matrix product for fast multiplication of (small) stacked matrices."""

    row, col = in_a.shape

    for j in range(col):
        for i in range(row):
            out[j] += in_a[i][j] * in_b[j]
            
            
class _iterator:
    """Utility routines to iterate over local neighbourhoods."""
    
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
    """Utility routines to evaluate interpolation kernel at a given point."""

    @staticmethod
    def _make_evaluate(kernel_type):
        
        if kernel_type == "NN":
            def _evaluate_1d(frame, sample_idx, kernel_idx, grid_offset):  
                
                # get index
                index = kernel_idx[0][frame][sample_idx]
                
                return index
            
            def _evaluate_2d(frame, sample_idx, kernel_idx, grid_offset):  
                                        
                # get index
                index = kernel_idx[0][frame][sample_idx] + \
                        kernel_idx[1][frame][sample_idx] * grid_offset[0]
                        
                return index
            
            def _evaluate_3d(frame, sample_idx, kernel_idx, grid_offset):  
                        
                # get index
                index = kernel_idx[0][frame][sample_idx] + \
                        kernel_idx[1][frame][sample_idx] * grid_offset[0] + \
                        kernel_idx[2][frame][sample_idx] * grid_offset[1]
                        
                return index
            
        elif kernel_type == "KB":
            def _evaluate_1d(frame, sample_idx, neighbour_idx, kernel_value, kernel_idx, grid_offset):  
                
                # get value
                value = kernel_value[0][frame][sample_idx][neighbour_idx[0]]
                
                # get index
                index = kernel_idx[0][frame][sample_idx][neighbour_idx[0]]
                
                return value, index
            
            def _evaluate_2d(frame, sample_idx, neighbour_idx, kernel_value, kernel_idx, grid_offset):  
                
                # get value
                value = kernel_value[0][frame][sample_idx][neighbour_idx[0]] * \
                        kernel_value[1][frame][sample_idx][neighbour_idx[1]]
                        
                # get index
                index = kernel_idx[0][frame][sample_idx][neighbour_idx[0]] + \
                        kernel_idx[1][frame][sample_idx][neighbour_idx[1]] * grid_offset[0]
                        
                return value, index
            
            def _evaluate_3d(frame, sample_idx, neighbour_idx, kernel_value, kernel_idx, grid_offset):  
                
                # get value
                value = kernel_value[0][frame][sample_idx][neighbour_idx[0]] * \
                        kernel_value[1][frame][sample_idx][neighbour_idx[1]] * \
                        kernel_value[2][frame][sample_idx][neighbour_idx[2]]
                        
                # get index
                index = kernel_idx[0][frame][sample_idx][neighbour_idx[0]] + \
                        kernel_idx[1][frame][sample_idx][neighbour_idx[1]] * grid_offset[0] + \
                        kernel_idx[2][frame][sample_idx][neighbour_idx[2]] * grid_offset[1]
                        
                return value, index
        
        return _evaluate_1d, _evaluate_2d, _evaluate_3d

    
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
            idx_in = (coeff, batch, index_in)

            # get total weight
            weight = kernel_value * basis_adjoint[frame, coeff]

            # update data
            data_out[idx_out] += weight * data_in[idx_in]
