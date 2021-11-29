# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:15:00 2021

@author: mcencini
"""
# pylint: disable=no-member

from typing import List, Tuple, Dict, Union

import numpy as np

import torch
from torch import Tensor


def FFT():
    pass


def prepareNUFFT():
    pass


def NUFFT():
    pass


def NUFFTAdjoint():
    pass


def prepareNUFFTSelfAdjoint():
    pass


def NUFFTSelfAdjoint():
    pass


class DeviceDispatch:
    """Manage computational devices."""

    computation_device: str
    data_device: str

    def __init__(self, computation_device: str, data_device: str):
        """DeviceDispatch object constructor.

        Args:
            computation_device: target device to perform the computation.
            data_device: original device hosting the data (could be the same)
                         as computation device.

        """
        self.computation_device = computation_device
        self.data_device = data_device

    def dispatch(self, *tensors):
        """Dispatch input to computational device."""
        for tensor in tensors:
            tensor.to(self.computation_device)

    def gather(self, *tensors):
        """Gather output to original device"""
        for tensor in tensors:
            tensor.to(self.data_device)


class DataReshape:
    """Ravel and unravel multi-channel/-echo/-slice sparse k-space data."""
    batch_shape: Union[List[int], Tuple[int]]
    batch_size: int
    nframes: int
    batch_axis_shape: Union[None, List[int], Tuple[int]]

    def __init__(self, coord_shape: Union[List[int], Tuple[int]]):
        """DataReshape object constructor.

        Args:
            coord_shape: shape of k-space coordinates tensor.

        """
        self.batch_shape = coord_shape[1:-1]
        self.batch_size = np.prod(self.batch_shape)
        self.nframes = coord_shape[0]

    def ravel(self, input: Tensor) -> Tensor:
        """Ravel multi-channel/-echo/-slice data.

        Args:
            input (tensor): input data tensor of size [*(batch_axis_shape), *(batch_shape)]
                            where batch_axis are one or more dimensions corresponding
                            to different batches (e.g. channels/echoes/slices/...)
                            with same spectral locations
                            and batch_shape are one or more dimension describing
                            different spectral locations 
                            (e.g. (n_readouts, n_point_per readout)).

        Returns:
            tensor: output 3D raveled data tensor of shape 
                    (n_frames, n_batches, n_spectral_locations).
        """
        # keep original batch shape
        self.batch_axis_shape = input.shape[1:-len(self.batch_shape)]
        nbatches = np.prod(self.batch_axis_shape)

        return input.reshape(self.nframes, nbatches, self.batch_size)

    def unravel(self, input: Tensor) -> Tensor:
        """Unavel raveled data, restoring original shape.

        Args:
            input (tensor): input 3D raveled data tensor  of shape 
                            (n_frames, n_batches, n_spectral_locations).

        Returns:
            tensor: output original-shape data.
        """
        return input.reshape(self.nframes, *self.batch_axis_shape, *self.batch_shape)


class GriddedReshape:
    """Ravel and unravel multi-channel/-echo/-slice gridded k-space data."""
    ndim: int
    batch_axis_shape: Union[None, List[int], Tuple[int]]

    def __init__(self, coord_shape: Union[List[int], Tuple[int]]):
        """GriddedReshape object constructor.

        Args:
            coord_shape: shape of k-space coordinates tensor.

        """
        self.ndim = coord_shape[-1]
        self.nframes = coord_shape[0]
        self.grid_shape: Union[List[int], Tuple[int]]

    def ravel(self, input: Tensor) -> Tensor:
        """Ravel multi-channel/-echo/-slice data.

        Args:
            input (tensor): input data tensor of size [*(batch_axis_shape), *(grid_shape)]
                            where batch_axis are one or more dimensions corresponding
                            to different batches (e.g. channels/echoes/slices/...)
                            with same spectral locations
                            and batch_shape are one or more dimension describing
                            different spectral locations 
                            (e.g. ((nz), ny, nx)).

        Returns:
            tensor: output 3D raveled data tensor of shape 
                    (n_frames, n_batches, n_spectral_locations).
        """
        # keep original batch shape
        self.batch_axis_shape = input.shape[1:-self.ndim]
        self.grid_shape = input.shape[-self.ndim:]

        nbatches = np.prod(self.batch_axis_shape)
        batch_size = np.prod(self.grid_shape)

        return input.reshape(self.nframes, nbatches, batch_size)

    def unravel(self, input: Tensor) -> Tensor:
        """Unavel raveled data, restoring original shape.

        Args:
            input (tensor): input 3D raveled data tensor  of shape 
                            (n_frames, n_batches, n_spectral_locations).

        Returns:
            tensor: output original-shape data.
        """
        return input.reshape(self.nframes, *self.batch_axis_shape, *self.grid_shape)


class Apodization:
    """Image-domain apodization operator to correct effect of convolution."""
    ndim: int
    apod: List[float]

    def __init__(self,
                 grid_shape: Union[List[int], Tuple[int]],
                 oversamp: float,
                 width: Union[List[int], Tuple[int]],
                 beta: Union[List[float], Tuple[float]]):

        # get number of spatial dimensions
        self.ndim = len(grid_shape)
        self.apod = []

        for axis in range(-self.ndim, 0):
            i = grid_shape[axis]
            os_i = np.ceil(oversamp * i)
            idx = torch.arange(i, dtype=torch.float32)

            # Calculate apodization
            apod = (beta[axis]**2 - (np.pi * width[axis]
                    * (idx - i // 2) / os_i)**2)**0.5
            apod /= torch.sinh(apod)
            self.apod.append(apod.reshape([i] + [1] * (-axis - 1)))

    def apply(self, input: Tensor):
        """Apply apodization step in-place.

        Args:
            input (tensor): Input image.
        """
        for axis in range(-self.ndim, 0):
            input *= self.apod[axis]


def _get_oversampled_shape(oversamp: float, shape:  Union[List[int], Tuple[int]]) -> Tuple[int]:
    return [np.ceil(oversamp * axis).astype(int) for axis in shape]


class ZeroPadding:
    """ Image-domain padding operator to interpolate k-space on oversampled grid."""
    padsize: Tuple[int]

    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]):
        # get oversampled shape
        oshape = _get_oversampled_shape(oversamp, shape)
        
        # get pad size
        padsize = [(oshape[axis] - shape[axis]) // 2 for axis in range(len(shape))]
        
        # get amount of pad over each direction
        pad = np.repeat(padsize, 2)
        pad.reverse() # torch take from last to first
        
        self.padsize = tuple(pad)

    def apply(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        return torch.nn.functional.pad(input, self.padsize, mode="constat", value=0)


class Cropping:
    """Image-domain cropping operator to select targeted FOV."""
    cropsize: Tuple[int]

    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]):
        # get oversampled shape
        oshape = _get_oversampled_shape(oversamp, shape)
        
        # get crop size
        cropsize = [(shape[axis] - oshape[axis]) // 2 for axis in range(len(shape))]
        
        # get amount of crop over each direction
        crop = np.repeat(cropsize, 2)
        crop.reverse() # torch take from last to first
        
        self.cropsize = crop

    def apply(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        return torch.nn.functional.pad(input, self.cropsize)


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


# def interpolate(data_in: Tensor, sparse_coeff: Dict, adjoint_basis: Union[None, Tensor]) -> Tensor:
#     """Interpolation from array to points specified by coordinates.

#     Args:
#         data_in (tensor): Input Cartesian array.
#         sparse_coeff (dict): pre-calculated interpolation coefficients in sparse COO format.
#         adjoint_basis (tensor): Adjoint low rank subspace projection operator (subspace to time); can be None.

#     Returns:
#         data_out (tensor): Output Non-Cartesian array.
#     """
#     # unpack input
#     index = sparse_coeff['index']
#     value = sparse_coeff['value']
#     shape = sparse_coeff['shape']
#     pts_shape = sparse_coeff['pts_shape']
#     ndim = sparse_coeff['ndim']
#     device = sparse_coeff['device']

#     # get input sizes
#     nframes = index[0].shape[0]
#     npts = _util.prod(pts_shape)

#     # reformat data for computation
#     batch_shape = data_in.shape[1:-ndim]
#     batch_size = _util.prod(batch_shape)  # ncoils * nslices * [int]

#     data_in = data_in.reshape([data_in.shape[0], batch_size, *shape])

#     # preallocate output data
#     data_out = torch.zeros((nframes, batch_size, npts),  # pylint: disable=no-member
#                            dtype=data_in.dtype, device=data_in.device)

#     # do actual interpolation
#     if device == 'cpu':
#         do_interpolation[ndim-2](data_out, data_in,
#                                  value, index, adjoint_basis)
#     else:
#         do_interpolation_cuda[ndim-2](
#             data_out, data_in, value, index, adjoint_basis)

#     # reformat for output
#     data_out = data_out.reshape([nframes, *batch_shape, *pts_shape])

#     return data_out


# def gridding(data_in: Tensor, sparse_coeff: Dict,  basis: Union[None, Tensor]) -> Tensor:
#     """Gridding of points specified by coordinates to array.

#     Args:
#         data_in (tensor): Input Non-Cartesian array.
#         sparse_coeff (dict): pre-calculated interpolation coefficients in sparse COO format.
#         basis (tensor): Low rank subspace projection operator (time to subspace); can be None.

#     Returns:
#         data_out (tensor): Output Cartesian array.
#     """
#     # unpack input
#     index = sparse_coeff['index']
#     value = sparse_coeff['value']
#     shape = sparse_coeff['shape']
#     pts_shape = sparse_coeff['pts_shape']
#     ndim = sparse_coeff['ndim']
#     device = sparse_coeff['device']

#     # get input sizes
#     nframes = index[0].shape[0]
#     npts = _util.prod(pts_shape)

#     # get number of coefficients
#     if basis is not None:
#         ncoeff = basis.shape[0]
#     else:
#         ncoeff = nframes

#     # reformat data for computation
#     batch_shape = data_in.shape[1:-len(pts_shape)]
#     batch_size = _util.prod(batch_shape)  # ncoils * nslices * [int]

#     # argument reshape
#     data_in = data_in.reshape([nframes, batch_size, npts])

#     # preallocate output data
#     data_out = torch.zeros((ncoeff, batch_size, *shape),  # pylint: disable=no-member
#                            dtype=data_in.dtype, device=data_in.device)

#     # do actual gridding
#     if device == 'cpu':
#         do_gridding[ndim-2](data_out, data_in, value, index, basis)
#     else:
#         do_gridding_cuda[ndim-2](
#             data_out, data_in, value, index, basis)

#     # reformat for output
#     data_out = data_out.reshape([ncoeff, *batch_shape, *shape])

#     return data_out
