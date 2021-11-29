# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:15:00 2021

@author: mcencini
"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=redefined-builtin

from typing import List, Tuple, Dict, Union

import numpy as np

import torch
from torch import Tensor, device


def fft():
    """Cartesian FFT."""
    pass


def ifft():
    """Cartesian Inverse FFT."""
    pass


def prepare_nufft(coord: Tensor,
                  shape: Union[int, List[int], Tuple[int]],
                  oversamp: Union[float, List[float], Tuple[float]] = 1.125,
                  width: Union[int, List[int], Tuple[int]] = 3,
                  basis: Union[None, Tensor] = None,
                  device: str = 'cpu') -> Dict:
    """Precompute nufft object for faster t_nufft / t_nufft_adjoint.
    Args:
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        shape (int or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.

    Returns:
        dict: structure containing interpolator object.
    """
    # get parameters
    ndim = coord.shape[-1]

    if np.isscalar(width):
        width = np.array(  # pylint: disable=no-member
            [width] * ndim, dtype=np.int16)  # pylint: disable=no-member
    else:
        width = np.array(  # pylint: disable=no-member
            width, dtype=np.int16)  # pylint: disable=no-member

    # calculate Kaiser-Bessel beta parameter
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    if np.isscalar(shape):
        shape = np.array(  # pylint: disable=no-member
            [shape] * ndim, dtype=np.int16)  # pylint: disable=no-member
    else:
        shape = np.array(  # pylint: disable=no-member
            shape, dtype=np.int16)  # pylint: disable=no-member

    os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

    # adjust coordinates
    coord = _util.scale_coord(coord, shape, oversamp)

    # compute interpolator
    sparse_coeff = _interp.prepare_interpolator(
        coord, os_shape, width, beta, device)

    # set basis
    if basis is not None:
        basis = basis.to(device)

    return {'ndim': ndim,
            'oversamp': oversamp,
            'width': width,
            'beta': beta,
            'os_shape': os_shape,
            'oshape': shape,
            'sparse_coeff': sparse_coeff,
            'device': device,
            'basis': basis}


def nufft(image: Tensor, interpolator: Dict) -> Tensor:
    """Non-uniform Fast Fourier Transform.
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    sparse_coeff = interpolator['sparse_coeff']
    device = interpolator['device']

    # Copy input to avoid original data modification
    image = image.clone()

    # Original device
    odevice = image.device

    # Offload to computational device
    image = image.to(device)

    # Apodize
    Apodize(ndim, oversamp, width, beta)(image)

    # Zero-pad
    image = ZeroPad(oversamp, image.shape[-ndim:])(image)

    # FFT
    kdata = FFT()(image, axes=range(-ndim, 0), norm='ortho')

    # Interpolate
    kdata = DeGrid(sparse_coeff)(image)

    # Bring back to original device
    kdata = kdata.to(odevice)

    return kdata


def nufft_adjoint(kdata: Tensor, interpolator: Dict) -> Tensor:
    """Adjoint Non-uniform Fast Fourier Transform.
    """
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    sparse_coeff = interpolator['sparse_coeff']
    device = interpolator['device']

    # Original device
    odevice = kdata.device

    # Offload to computational device
    kdata = kdata.to(device)

    # Gridding
    kdata = Grid(sparse_coeff)(kdata)

    # IFFT
    image = IFFT()(kdata, axes=range(-ndim, 0), norm='ortho')

    # Crop
    image = Crop(oversamp, image.shape[:-ndim])(image)

    # Apodize
    Apodize(ndim, oversamp, width, beta)(image)

    # Bring back to original device
    kdata = kdata.to(odevice)
    image = image.to(odevice)

    return image


# def prepare_nufft_selfadjoint(coord: Tensor,
#                      shape: Union[int, List[int], Tuple[int]],
#                      oversamp: Union[float, List[float], Tuple[float]] = 1.125,
#                      width: Union[int, List[int], Tuple[int]] = 3,
#                      basis: Tensor = None,
#                      device: str = 'cpu',
#                      dcf: Tensor = None) -> Dict:
#     """Compute spatio-temporal kernel for fast self-adjoint operation.
#     Args:
#         coord (tensor): Coordinate array of shape [nframes, ..., ndim]
#         shape (list or tuple of ints): Overesampled grid size.
#         width (list or tuple of int): Interpolation kernel full-width.
#         beta (list or tuple of floats): Kaiser-Bessel beta parameter.
#         device (str): identifier of computational device used for interpolation.
#         basis (tensor): low-rank temporal subspace basis.
#         dcf (tensor): k-space sampling density compensation weights.

#     Returns:
#         toeplitz (tensor): Fourier transform of system transfer Point Spread Function
#                            (spatiotemporal kernel)
#     """
#     # get initial data shape
#     nframes = coord.shape[0]
#     pts_shape = coord.shape[1:-1]
#     npts = _util.prod(pts_shape)
#     ndim = coord.shape[-1]

#     # reshape input
#     coord = coord.reshape([nframes, npts, ndim])

#     if np.isscalar(width):
#         width = np.array(  # pylint: disable=no-member
#             [width] * ndim, dtype=np.int16)  # pylint: disable=no-member
#     else:
#         width = np.array(  # pylint: disable=no-member
#             width, dtype=np.int16)  # pylint: disable=no-member

#     # calculate Kaiser-Bessel beta parameter
#     beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

#     if np.isscalar(shape):
#         shape = np.array(  # pylint: disable=no-member
#             [shape] * ndim, dtype=np.int16)  # pylint: disable=no-member
#     else:
#         shape = np.array(  # pylint: disable=no-member
#             shape, dtype=np.int16)  # pylint: disable=no-member

#     # get oversampled grid
#     os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

#     # scale coordinates
#     coord = _util.scale_coord(coord, shape, oversamp)

#     # actual kernel precomputation
#     kernel = _interp.prepare_toeplitz(coord,
#                                       os_shape,
#                                       width=width,
#                                       beta=beta,
#                                       basis=basis,
#                                       device=device,
#                                       dcf=dcf)

#     return {'kernel': kernel, 'oversamp': oversamp, 'ndim': ndim, 'device': device}


# def nufft_selfadjoint(image: Tensor, toeplitz: Dict) -> Tensor:
#     """Perform in-place fast self-adjoint by convolution with spatio-temporal kernel matrix.

#     Args:
#         data_in (tensor): Input image.
#         toeplitz (dict): Pre-computed spatio-temporal kernel.

#     Returns
#         data_out (tensor): Output image.

#     """
#     # unpack input
#     toeplitz_kernel = toeplitz['kernel']
#     oversamp = toeplitz['oversamp']
#     ndim = toeplitz['ndim']
#     islowrank = toeplitz_kernel['islowrank']
#     device = toeplitz['device']

#     # Original device
#     odevice = image.device

#     # Offload to computational device
#     image = image.to(device)

#     ishape = list(image.shape)
#     image = image.reshape(
#         [ishape[0], _util.prod(ishape[1:-ndim]), *ishape[-ndim:]])

#     # Get oversampled shape
#     shape = list(image.shape)
#     os_shape = _util.get_oversamp_shape(shape, oversamp, ndim)

#     # Zero-pad
#     image = _util.resize(image, os_shape)

#     # FFT
#     data_in = fft(image, axes=range(-ndim, 0), center=False, norm=None)

#     # Reshape input
#     if islowrank is True:
#         data_in = data_in.reshape(
#             [shape[0], shape[1], _util.prod(os_shape[-ndim:])])

#     # Preallocate output
#     data_out = torch.zeros(  # pylint: disable=no-member
#         data_in.shape, dtype=data_in.dtype, device=data_in.device)

#     # Perform convolution
#     _interp.toeplitz(data_out, data_in, toeplitz_kernel)

#     # Reshape output
#     if islowrank is True:
#         data_out = data_out.reshape(os_shape)

#     # IFFT
#     image = ifft(data_out, axes=range(-ndim, 0), center=False, norm=None)

#     # Crop
#     image = _util.resize(image, shape)
#     image = image.reshape(ishape)

#     if islowrank is True:
#         image = image[:, 0]

#     # Bring back to original device
#     image = image.to(odevice)

#     return image


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


class Apodize:
    """Image-domain apodization operator to correct effect of convolution."""
    ndim: int
    apod: List[float]

    def __init__(self,
                 grid_shape: Union[List[int], Tuple[int]],
                 oversamp: float,
                 width: Union[List[int], Tuple[int]],
                 beta: Union[List[float], Tuple[float]],
                 device: Union[str, device]):

        # get number of spatial dimensions
        self.ndim = len(grid_shape)
        self.apod = []

        for axis in range(-self.ndim, 0):
            i = grid_shape[axis]
            os_i = np.ceil(oversamp * i)
            idx = torch.arange(i, dtype=torch.float32, device=device)

            # Calculate apodization
            apod = (beta[axis]**2 - (np.pi * width[axis] * (idx - i // 2) / os_i)**2)**0.5
            apod /= torch.sinh(apod)
            self.apod.append(apod.reshape([i] + [1] * (-axis - 1)))

    def __call__(self, input: Tensor):
        """Apply apodization step in-place.

        Args:
            input (tensor): Input image.
        """
        for axis in range(-self.ndim, 0):
            input *= self.apod[axis]


def _get_oversampled_shape(oversamp: float, shape:  Union[List[int], Tuple[int]]) -> Tuple[int]:
    return [np.ceil(oversamp * axis).astype(int) for axis in shape]


class ZeroPad:
    """ Image-domain padding operator to interpolate k-space on oversampled grid."""
    padsize: Tuple[int]

    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]):
        # get oversampled shape
        oshape = _get_oversampled_shape(oversamp, shape)

        # get pad size
        padsize = [(oshape[axis] - shape[axis]) //
                   2 for axis in range(len(shape))]

        # get amount of pad over each direction
        pad = np.repeat(padsize, 2)
        pad.reverse()  # torch take from last to first

        self.padsize = tuple(pad)

    def __call__(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        return torch.nn.functional.pad(input, self.padsize, mode="constat", value=0)
    
    
class Crop:
    """Image-domain cropping operator to select targeted FOV."""
    cropsize: Tuple[int]

    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]):
        # get oversampled shape
        oshape = _get_oversampled_shape(oversamp, shape)

        # get crop size
        cropsize = [(shape[axis] - oshape[axis]) //
                    2 for axis in range(len(shape))]

        # get amount of crop over each direction
        crop = np.repeat(cropsize, 2)
        crop.reverse()  # torch take from last to first

        self.cropsize = crop

    def __call__(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        return torch.nn.functional.pad(input, self.cropsize)
 
    
def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))
    
class FFT:
    """Cartesian Fourier Transform."""
    
    def __call__(input, oshape=None, axes=None, center=True, norm='ortho'):
        # allow for single scalar axis
        if np.isscalar(axes):
            axes = [axes]

        # transform to list to allow range object
        axes = list(axes)

        if center:
            output = FFT._fftc(input, oshape=oshape, axes=axes, norm=norm)
        else:
            output = torch.fft.fftn(input, s=oshape, dim=axes, norm=norm)

        if torch.is_complex(input) and input.dtype != output.dtype:
            output = output.to(input.dtype, copy=False)

        return output

    @staticmethod
    def _fftc(input, oshape=None, axes=None, norm='ortho'):
        
        # adapt output shape
        if oshape is None:
            oshape = input.shape
        else:
            oshape = (*input.shape[:-len(oshape)], *oshape)

        # process axes arg
        ndim = input.ndim
        axes = _normalize_axes(axes, ndim)

        # actual fft
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.fftn(tmp, s=oshape, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

        return output

class IFFT:
    """Cartesian Inverse Fourier Transform."""
    
    def __call__(input, oshape=None, axes=None, center=True, norm='ortho'):
        # allow for single scalar axis
        if np.isscalar(axes):
            axes = [axes]

        # transform to list to allow range object
        axes = list(axes)

        if center:
            output = IFFT._ifftc(input, oshape=oshape, axes=axes, norm=norm)
        else:
            output = torch.fft.ifftn(input, s=oshape, dim=axes, norm=norm)

        if torch.is_complex(input) and input.dtype != output.dtype:
            output = output.to(input.dtype, copy=False)

        return output

    @staticmethod
    def _ifftc(input, oshape=None, axes=None, norm='ortho'):
        
        # adapt output shape
        if oshape is None:
            oshape = input.shape
        else:
            oshape = (*input.shape[:-len(oshape)], *oshape)

        # process axes arg
        ndim = input.ndim
        axes = _normalize_axes(axes, ndim)

        # actual fft
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.ifftn(tmp, s=oshape, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

        return output

class Grid:
    """K-space data gridding and de-gridding operator."""

    def __init__(self, kernel_dict: Dict):
        pass

    def __call__(self, input: Tensor) -> Tensor:
        """Grid sparse k-space data on a Cartesian Grid.

        Args:
            input (tensor): Cartesian k-space data.

        Returns:
            tensor: Non-Cartesian sparse k-space data.
        """
        pass

class DeGrid:
    
    def __init__(self, kernel_dict: Dict):
        pass
    
    def __call__(self, input: Tensor) -> Tensor:
        """Interpolate Cartesian k-space data to given coordinates.

        Args:
            input (tensor): Non-Cartesian sparse k-space data.

        Returns:
            tensor: Output Cartesian k-space data.
        """
        pass
