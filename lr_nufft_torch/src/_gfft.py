# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:15:00 2021

@author: mcencini
"""
from typing import List, Tuple, Dict, Union

import torch
from torch import Tensor

class DeviceDispatch:
    """Manage computational devices."""
    def __init__(self, device_id: str):
        pass
    
    def dispatch(self, input: Tensor) -> Tensor:
        """Dispatch input to computational device."""
        pass
    
    def gather(self, input: Tensor) -> Tensor:
        """Gather output to original device"""
        pass


class DataReshape:
    """Ravel and unravel multi-channel/-echo/-slice data."""
    def __init__(self, shape: Union[List[int], Tuple[int]]):
        pass
    
    def ravel(self, input: Tensor) -> Tensor:
        """Ravel multi-channel/-echo/-slice data.
        
        Args:
            input (tensor): input data tensor of size [(batch_axis), (data_shape)]
                            where batch_axis are one or more dimensions corresponding
                            to different batches (e.g. channels/echoes/slices/...)
                            with same spectral or spatial locations
                            and data_shape are one or more dimension describing
                            different spectral/spatial locations 
                            (e.g. (n_readouts, n_point_per readout) or (nz, ny, nx)).
        
        Returns:
            tensor: output 2D raveled data tensor of shape 
                    (n_batches, n_spectral_or_spatial_locations).
        """
        pass
    
    
    def unravel(self, input: Tensor) -> Tensor:
        """Unavel raveled data, restoring original shape.
        
        Args:
            input (tensor): input 2D raveled data tensor  of shape 
                            (n_batches, n_spectral_or_spatial_locations).
        
        Returns:
            tensor: output original-shape data.
        """
        pass
    

class Apodization:
    """Image-domain apodization operator to correct effect of convolution."""
    def __init__(self, 
                 oversamp: float, 
                 width: Union[List[int], Tuple[int]], 
                 beta: Union[List[float], Tuple[float]]):
        pass

    def apply(self, input: Tensor):
        """Apply apodization step in-place.

        Args:
            input (tensor): Input image.
        """
        pass


class ZeroPadding:
    """ Image-domain padding operator to interpolate k-space on oversampled grid."""
    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]) :
        pass

    def apply(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.
            
        Returns:
            tensor: Output zero-padded image.
        """
        pass


class Cropping:
    """Image-domain cropping operator to select targeted FOV."""
    def __init__(self, shape: Union[List[int], Tuple[int]]) :
        pass

    def apply(self, input: Tensor) -> Tensor:
        """ Apply cropping step.

        Args:
            input (tensor): Input image.
            
        Returns:
            tensor: Output cropped image.
        """
        pass


class Gridding:
    """K-space data gridding and de-gridding operator."""
    def __init__(self, kernel_dict: Dict):
        pass

    def grid_data(self, input: Tensor) -> Tensor:
        """ Apply cropping step.

        Args:
            input (tensor): Input image.
            
        Returns:
            tensor: Output cropped image.
        """
        pass
    
    def degrid_data(self, input: Tensor) -> Tensor:
        """ Apply cropping step.

        Args:
            input (tensor): Input image.
            
        Returns:
            tensor: Output cropped image.
        """
        pass


def FFT():
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
