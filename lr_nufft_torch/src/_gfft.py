# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:15:00 2021

@author: mcencini
"""
from typing import List, Tuple, Dict, Union

import torch


class DeviceDispatcher:
    pass

class Apodization:
    """ Image-domain apodization operator to correct effect of convolution."""

    def __init__(self, ndim: int, 
                 oversamp: float, 
                 width, beta):
        pass

    def apply(self, input):
        """ Apply apodization step.

        Args:
            input (tensor): Input image.
        """
        pass


class _ZeroPadding:
    pass


class _Cropping:
    pass


class _Gridding:
    pass


class _FFT:
    pass


def interpolate(data_in: Tensor, sparse_coeff: Dict, adjoint_basis: Union[None, Tensor]) -> Tensor:
    """Interpolation from array to points specified by coordinates.

    Args:
        data_in (tensor): Input Cartesian array.
        sparse_coeff (dict): pre-calculated interpolation coefficients in sparse COO format.
        adjoint_basis (tensor): Adjoint low rank subspace projection operator (subspace to time); can be None.

    Returns:
        data_out (tensor): Output Non-Cartesian array.
    """
    # unpack input
    index = sparse_coeff['index']
    value = sparse_coeff['value']
    shape = sparse_coeff['shape']
    pts_shape = sparse_coeff['pts_shape']
    ndim = sparse_coeff['ndim']
    device = sparse_coeff['device']

    # get input sizes
    nframes = index[0].shape[0]
    npts = _util.prod(pts_shape)

    # reformat data for computation
    batch_shape = data_in.shape[1:-ndim]
    batch_size = _util.prod(batch_shape)  # ncoils * nslices * [int]

    data_in = data_in.reshape([data_in.shape[0], batch_size, *shape])

    # preallocate output data
    data_out = torch.zeros((nframes, batch_size, npts),  # pylint: disable=no-member
                           dtype=data_in.dtype, device=data_in.device)

    # do actual interpolation
    if device == 'cpu':
        do_interpolation[ndim-2](data_out, data_in,
                                 value, index, adjoint_basis)
    else:
        do_interpolation_cuda[ndim-2](
            data_out, data_in, value, index, adjoint_basis)

    # reformat for output
    data_out = data_out.reshape([nframes, *batch_shape, *pts_shape])

    return data_out


def gridding(data_in: Tensor, sparse_coeff: Dict,  basis: Union[None, Tensor]) -> Tensor:
    """Gridding of points specified by coordinates to array.

    Args:
        data_in (tensor): Input Non-Cartesian array.
        sparse_coeff (dict): pre-calculated interpolation coefficients in sparse COO format.
        basis (tensor): Low rank subspace projection operator (time to subspace); can be None.

    Returns:
        data_out (tensor): Output Cartesian array.
    """
    # unpack input
    index = sparse_coeff['index']
    value = sparse_coeff['value']
    shape = sparse_coeff['shape']
    pts_shape = sparse_coeff['pts_shape']
    ndim = sparse_coeff['ndim']
    device = sparse_coeff['device']

    # get input sizes
    nframes = index[0].shape[0]
    npts = _util.prod(pts_shape)

    # get number of coefficients
    if basis is not None:
        ncoeff = basis.shape[0]
    else:
        ncoeff = nframes

    # reformat data for computation
    batch_shape = data_in.shape[1:-len(pts_shape)]
    batch_size = _util.prod(batch_shape)  # ncoils * nslices * [int]

    # argument reshape
    data_in = data_in.reshape([nframes, batch_size, npts])

    # preallocate output data
    data_out = torch.zeros((ncoeff, batch_size, *shape),  # pylint: disable=no-member
                           dtype=data_in.dtype, device=data_in.device)

    # do actual gridding
    if device == 'cpu':
        do_gridding[ndim-2](data_out, data_in, value, index, basis)
    else:
        do_gridding_cuda[ndim-2](
            data_out, data_in, value, index, basis)

    # reformat for output
    data_out = data_out.reshape([ncoeff, *batch_shape, *shape])

    return data_out
