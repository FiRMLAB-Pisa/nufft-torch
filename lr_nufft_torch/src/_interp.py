# -*- coding: utf-8 -*-
"""
Kaiser-Bessel interpolation subroutines.

Adapted from SigPy [1]. Compared to [1], we use a reduced memory
radix tensor decomposition of pre-computed interpolators, to speed-up
computation with respect to precomputation-free NUFFT [2,3] while
mantaining a reasonably light memory footprint in high-dimensional
problems (non-separable 3D+t data). We include interpolation
routines with embedded low-rank subspace projection operators for
model-based MRI reconstructions.

[1]: Ong, F., and M. Lustig. "SigPy: a python package for high performance iterative reconstruction."
         Proceedings of the ISMRM 27th Annual Meeting, Montreal, Quebec, Canada. Vol. 4819. 2019.
[2]: Lin, Jyh-Miin. "Python non-uniform fast Fourier transform (PyNUFFT):
     An accelerated non-Cartesian MRI package on a heterogeneous platform (CPU/GPU)."
     Journal of Imaging 4.3 (2018): 51.
[3]: Lin, Jyh-Miin, et al.
     "Memory reduced non-Cartesian MRI encoding using the mixed-radix tensor product on CPU and GPU."
     arXiv preprint arXiv:1903.08365 (2019).
[4]: McGivney DF, Pierre E, Ma D, et al.
     SVD compression for magnetic resonance fingerprinting in the time domain.
     IEEE Trans Med Imaging. 2014;33(12):2311-2322. doi:10.1109/TMI.2014.2337321

"""
from typing import List, Tuple, Dict, Union

import numpy as np
import numba as nb

import torch
from torch import Tensor

from lr_nufft_torch import _util

# fix this
# from numba import typed
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# CUDA settings
threadsperblock = 32

#%%
def prepare_interpolator(coord: Tensor,
                         shape: Union[List[int], Tuple[int]],
                         width: Union[List[int], Tuple[int]],
                         beta: Union[List[float], Tuple[float]],
                         device: str) -> Dict:
    """Precompute nufft object for faster t_nufft / t_nufft_adjoint.
    
    Args:
        coord (tensor): Coordinate array of shape [nframes, [int], ndim]
        shape (list or tuple of ints): Overesampled grid size.
        width (list or tuple of int): Interpolation kernel full-width.
        beta (list or tuple of floats): Kaiser-Bessel beta parameter.
        device (str): identifier of computational device used for interpolation.

    Returns:
        dict: structure containing sparse interpolator matrix.
    """
    # parse input sizes
    ndim = coord.shape[-1]
    nframes = coord.shape[0]
    pts_shape = coord.shape[1:-1]
    npts = _util.prod(pts_shape)

    # arg reshape
    coord = coord.reshape([nframes*npts, ndim]).T

    # preallocate interpolator
    # index = typed.List()
    # value = typed.List()
    index = []
    value = []

    for i in range(ndim):
        # kernel value
        value.append(torch.zeros(  # pylint: disable=no-member
            (nframes*npts, width[i]), dtype=torch.float32))  # pylint: disable=no-member
        # kernel index
        index.append(torch.zeros(  # pylint: disable=no-member
            (nframes*npts, width[i]), dtype=torch.int32))  # pylint: disable=no-member

    # actual precomputation
    for i in range(ndim):
        _do_prepare_interpolator(value[i], index[i],
                                 coord[i], width[i], beta[i], shape[i])

    # reformat for output
    for i in range(ndim):
        index[i] = index[i].reshape([nframes, npts, width[i]]).to(device)
        value[i] = value[i].reshape([nframes, npts, width[i]]).to(device)

    # get device identifier
    if device == 'cpu':
        device_str = device
    else:
        device_str = device[:4]  # cuda instead of cuda:n

    return {'index': index, 'value': value, 'shape': shape, 'pts_shape': pts_shape, 'ndim': ndim, 'device': device_str}


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


def prepare_toeplitz(coord: Tensor,
                     shape: Union[List[int], Tuple[int]],
                     width: Union[List[int], Tuple[int]],
                     beta: Union[List[float], Tuple[float]],
                     device: str,
                     basis: Tensor,
                     dcf: Tensor) -> Dict:
    """Compute spatio-temporal kernel for fast self-adjoint operation.
    
    Args:
        coord (tensor): Coordinate array of shape [nframes, [int], ndim]
        shape (list or tuple of ints): Overesampled grid size.
        width (list or tuple of int): Interpolation kernel full-width.
        beta (list or tuple of floats): Kaiser-Bessel beta parameter.
        device (str): identifier of computational device used for interpolation.
        basis (tensor): low-rank temporal subspace basis.
        dcf (tensor): k-space sampling density compensation weights.

    Returns:
        st_kernel (tensor): Fourier transform of system transfer Point Spread Function
                            (spatiotemporal kernel)
    """
    # get dimensions
    ndim = coord.shape[-1]
    npts = coord.shape[-2]

    # if dcf are not provided, assume uniform sampling density
    if dcf is None:
        dcf = torch.ones(coord.shape[:-1], torch.float32,  # pylint: disable=no-member
                         device=device)  # pylint: disable=no-member
    else:
        dcf = dcf.to(device)

    if dcf.ndim > 1 and basis is not None:
        dcf = dcf[:, None, :]

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        islowrank = True
        isreal = not torch.is_complex(basis)  # pylint: disable=no-member
        ncoeff, nframes = basis.shape
        basis = basis.to(device)
        adjoint_basis = basis.conj().T.to(device)

    else:
        islowrank = False
        isreal = False
        nframes, ncoeff = coord.shape[0], coord.shape[0]

    if isreal:
        dtype = torch.float32  # pylint: disable=no-member
    else:
        dtype = torch.complex64  # pylint: disable=no-member

    if basis is not None:
        basis = basis.to(dtype)
        adjoint_basis = adjoint_basis.to(dtype)

    if basis is not None:
        # initialize temporary arrays
        delta = torch.ones((nframes, ncoeff, npts),  # pylint: disable=no-member
                           dtype=torch.complex64, device=device)  # pylint: disable=no-member
        delta = delta * adjoint_basis[:, :, None]

    else:
        # initialize temporary arrays
        delta = torch.ones(  # pylint: disable=no-member
            (nframes, npts), dtype=torch.complex64, device=device)  # pylint: disable=no-member

    # calculate interpolator
    interpolator = prepare_interpolator(coord, shape, width, beta, device)
    st_kernel = gridding(delta, interpolator, basis)

    # keep only real part if basis is real
    if isreal:
        st_kernel = st_kernel.real

    # fftshift kernel to accelerate computation
    st_kernel = torch.fft.ifftshift(st_kernel, dim=list(range(-ndim, 0)))

    if basis is not None:
        st_kernel = st_kernel.reshape(
            [*st_kernel.shape[:2], _util.prod(st_kernel.shape[2:])])
    else:
        st_kernel = st_kernel[:, None, [int]]

    # normalize
    st_kernel /= torch.quantile(st_kernel, 0.95)  # pylint: disable=no-member

    # remove NaN
    st_kernel = torch.nan_to_num(st_kernel)  # pylint: disable=no-member

    # get device identifier
    if device == 'cpu':
        device_str = device
    else:
        device_str = device[:4]  # cuda instead of cuda:n

    return {'value': st_kernel, 'islowrank': islowrank, 'device': device_str}


def toeplitz(data_out: Tensor, data_in: Tensor, toeplitz_kernel: Dict) -> Tensor:
    """Perform in-place fast self-adjoint by multiplication in k-space with spatio-temporal kernel.
    
    Args:
        data_out (tensor): Output tensor of oversampled gridded k-space data.
        data_in (tensor): Input tensor of oversampled gridded k-space data.
        st_kernel (dict): Fourier transform of system transfer Point Spread Function
    """
    if toeplitz_kernel['islowrank'] is True:
        if toeplitz_kernel['device'] == 'cpu':
            do_selfadjoint_interpolation(data_out, data_in, toeplitz_kernel['value'])
        else:
            do_selfadjoint_interpolation_cuda(
                data_out, data_in, toeplitz_kernel['value'])
    else:
        data_out = toeplitz_kernel['value'] * data_in

# %% CPU specific routines


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _kaiser_bessel_kernel(x, beta):
    if abs(x) > 1:
        return 0

    x = beta * (1 - x**2)**0.5
    t = x / 3.75
    if x < 3.75:
        return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    else:
        return x**-0.5 * np.exp(x) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    row, col = in_a.shape

    for j in range(col):
        for i in range(row):
            out[j] += in_a[i][j] * in_b[j]

    return out


def _get_prepare_interpolator():
    """Subroutines for interpolator planning."""
    kernel = _kaiser_bessel_kernel

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _prepare_interpolator(interp_value, interp_index, coord,
                              kernel_width, kernel_param, grid_shape):

        # get sizes
        npts = coord.shape[0]
        kernel_width = interp_index.shape[-1]

        for i in nb.prange(npts):  # pylint: disable=not-an-iterable
            x_0 = np.ceil(coord[i] - kernel_width / 2)

            for x_i in range(kernel_width):
                val = kernel(
                    ((x_0 + x_i) - coord[i]) / (kernel_width / 2), kernel_param)

                # save interpolator
                interp_value[i, x_i] = val
                interp_index[i, x_i] = (x_0 + x_i) % grid_shape

    return _prepare_interpolator


_prepare_interpolator = _get_prepare_interpolator()

def _do_prepare_interpolator(interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape):
    """ Preparation routine wrapper. """
    interp_value = _util.pytorch2numba(interp_value)
    interp_index = _util.pytorch2numba(interp_index)
    coord = _util.pytorch2numba(coord)

    _prepare_interpolator(interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape)

    interp_value = _util.numba2pytorch(interp_value)
    interp_index = _util.numba2pytorch(interp_index, requires_grad=False)
    coord = _util.numba2pytorch(coord)


def _get_interpolate():
    """Subroutines for CPU time-domain interpolation (cartesian -> non-cartesian)."""

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _interpolate2(noncart_data, cart_data, interp_value, interp_index):

        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    noncart_data[frame, batch, point] += \
                        val * cart_data[frame, batch, idy, idx]

        return noncart_data

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _interpolate3(noncart_data, cart_data, interp_value, interp_index):

        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        noncart_data[frame, batch, point] += \
                            val * cart_data[frame, batch, idz, idy, idx]

        return noncart_data

    return _interpolate2, _interpolate3


def _get_interpolate_lowrank():
    """
    Subroutines for CPU low-rank interpolation.

    Transform cartesian low rank -> non-cartesian time domain.
    """

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _interpolate2(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while gathering data
                    for coeff in range(ncoeff):
                        noncart_data[frame, batch, point] += \
                            val * adjoint_basis[frame, coeff] * \
                            cart_data[coeff, batch, idy, idx]

        return noncart_data

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _interpolate3(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while gathering data
                        for coeff in range(ncoeff):
                            noncart_data[frame, batch, point] += \
                                val * adjoint_basis[frame, coeff] * \
                                cart_data[coeff, batch, idz, idy, idx]

        return noncart_data

    return _interpolate2, _interpolate3


def _do_interpolation2(data_out, data_in, value, index, adjoint_basis):
    """ 2D Interpolation routine wrapper. """
    data_out = _util.pytorch2numba(data_out)
    data_in = _util.pytorch2numba(data_in)
    value = [_util.pytorch2numba(val) for val in value]
    index = [_util.pytorch2numba(ind) for ind in index]

    if adjoint_basis is None:
        _get_interpolate()[0](data_out, data_in, value, index)
    else:
        adjoint_basis = _util.pytorch2numba(adjoint_basis)
        _get_interpolate_lowrank()[0](
            data_out, data_in, value, index, adjoint_basis)
        adjoint_basis = _util.numba2pytorch(adjoint_basis)

    data_out = _util.numba2pytorch(data_out)
    data_in = _util.numba2pytorch(data_in)
    value = [_util.numba2pytorch(val) for val in value]
    index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]


def _do_interpolation3(data_out, data_in, value, index, adjoint_basis):
    """ 3D Interpolation routine wrapper. """
    data_out = _util.pytorch2numba(data_out)
    data_in = _util.pytorch2numba(data_in)
    value = [_util.pytorch2numba(val) for val in value]
    index = [_util.pytorch2numba(ind) for ind in index]

    if adjoint_basis is None:
        _get_interpolate()[1](data_out, data_in, value, index)
    else:
        adjoint_basis = _util.pytorch2numba(adjoint_basis)
        _get_interpolate_lowrank()[1](
            data_out, data_in, value, index, adjoint_basis)
        adjoint_basis = _util.numba2pytorch(adjoint_basis)

    data_out = _util.numba2pytorch(data_out)
    data_in = _util.numba2pytorch(data_in)
    value = [_util.numba2pytorch(val) for val in value]
    index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]


do_interpolation = [_do_interpolation2, _do_interpolation3]


def _get_gridding():
    """Subroutines for CPU time-domain gridding (non-cartesian -> cartesian)."""

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _gridding2(cart_data, noncart_data, interp_value, interp_index):

        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames and batches
        for i in nb.prange(nframes*batch_size):  # pylint: disable=not-an-iterable

            # get current frame and batch index
            frame = i // batch_size
            batch = i % batch_size

            # iterate over non-cartesian point of current frame/batch
            for point in range(npts):

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        cart_data[frame, batch, idy, idx] += \
                            val * noncart_data[frame, batch, point]

        return cart_data

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _gridding3(cart_data, noncart_data, interp_value, interp_index):

        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames and batches
        for i in nb.prange(nframes*batch_size):  # pylint: disable=not-an-iterable

            # get current frame and batch index
            frame = i // batch_size
            batch = i % batch_size

            # iterate over non-cartesian point of current frame/batch
            for point in range(npts):

                # spread data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            cart_data[frame, batch, idz, idy, idx] += \
                                val * noncart_data[frame, batch, point]

        return cart_data

    return _gridding2, _gridding2


def _get_gridding_lowrank():
    """Subroutines for CPU low-rank gridding (non-cartesian time domain -> cartesian low-rank)."""

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _gridding2(cart_data, noncart_data, interp_value, interp_index, basis):

        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over low-rank coefficients and batches
        for i in nb.prange(ncoeff*batch_size):  # pylint: disable=not-an-iterable

            # get current low-rank coefficient and batch index
            coeff = i // batch_size
            batch = i % batch_size

            # iterate over frames in current coefficient/batch
            for frame in range(nframes):

                # iterate over non-cartesian point of current frame
                for point in range(npts):

                    # spread data within kernel radius
                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            # do adjoint low rank projection (low-rank subspace -> time domain)
                            # while spreading data
                            cart_data[coeff, batch, idy, idx] += \
                                val * basis[coeff, frame] * \
                                noncart_data[frame, batch, point]

        return cart_data

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _gridding3(cart_data, noncart_data, interp_value, interp_index, basis):

        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over low-rank coefficients and batches
        for i in nb.prange(ncoeff*batch_size):  # pylint: disable=not-an-iterable

            # get current low-rank coefficient and batch index
            coeff = i // batch_size
            batch = i % batch_size

            # iterate over frames in current coefficient/batch
            for frame in range(nframes):

                # iterate over non-cartesian point of current frame
                for point in range(npts):

                    # spread data within kernel radius
                    for i_z in range(zwidth):
                        idz = zindex[frame, point, i_z]
                        valz = zvalue[frame, point, i_z]

                        for i_y in range(ywidth):
                            idy = yindex[frame, point, i_y]
                            valy = valz * yvalue[frame, point, i_y]

                            for i_x in range(xwidth):
                                idx = xindex[frame, point, i_x]
                                val = valy * xvalue[frame, point, i_x]

                                # do adjoint low rank projection (low-rank subspace -> time domain)
                                # while gathering data
                                cart_data[coeff, batch, idz, idy, idx] += \
                                    val * basis[coeff, frame] * \
                                    noncart_data[frame, batch, point]

        return cart_data

    return _gridding2, _gridding3


def _do_gridding2(data_out, data_in, value, index, basis):
    """ 2D Gridding routine wrapper. """
    data_out = _util.pytorch2numba(data_out)
    data_in = _util.pytorch2numba(data_in)
    value = [_util.pytorch2numba(val) for val in value]
    index = [_util.pytorch2numba(ind) for ind in index]

    if basis is None:
        _get_gridding()[0](data_out, data_in, value, index)
    else:
        basis = _util.pytorch2numba(basis)
        _get_gridding_lowrank()[0](data_out, data_in, value, index, basis)
        basis = _util.numba2pytorch(basis)

    data_out = _util.numba2pytorch(data_out)
    data_in = _util.numba2pytorch(data_in)
    value = [_util.numba2pytorch(val) for val in value]
    index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]


def _do_gridding3(data_out, data_in, value, index, basis):
    """ 3D Gridding routine wrapper. """
    data_out = _util.pytorch2numba(data_out)
    data_in = _util.pytorch2numba(data_in)
    value = [_util.pytorch2numba(val) for val in value]
    index = [_util.pytorch2numba(ind) for ind in index]

    if basis is None:
        _get_gridding()[1](data_out, data_in, value, index)
    else:
        basis = _util.pytorch2numba(basis)
        _get_gridding_lowrank()[1](data_out, data_in, value, index, basis)
        basis = _util.numba2pytorch(basis)

    data_out = _util.numba2pytorch(data_out)
    data_in = _util.numba2pytorch(data_in)
    value = [_util.numba2pytorch(val) for val in value]
    index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]


do_gridding = [_do_gridding2, _do_gridding3]


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp_selfadjoint(data_out, data_in, toeplitz_matrix):

    n_coeff, batch_size, _ = data_in.shape

    for i in nb.prange(n_coeff * batch_size):  # pylint: disable=not-an-iterable
        coeff = i // batch_size
        batch = i % batch_size

        _dot_product(data_out[coeff][batch],
                     toeplitz_matrix[coeff], data_in[coeff][batch])

    return data_out


def do_selfadjoint_interpolation(data_out, data_in, toeplitz_matrix):
    """ Toeplitz routine wrapper. """
    data_out = _util.pytorch2numba(data_out)
    data_in = _util.pytorch2numba(data_in)
    toeplitz_matrix = _util.pytorch2numba(toeplitz_matrix)

    _interp_selfadjoint(data_out, data_in, toeplitz_matrix)

    data_out = _util.numba2pytorch(data_out)
    data_in = _util.numba2pytorch(data_in)
    toeplitz_matrix = _util.numba2pytorch(toeplitz_matrix)


# %% GPU specific routines
if torch.cuda.is_available():
    from numba import cuda

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_real(output, index, value):
        cuda.atomic.add(        # pylint: disable=too-many-function-args
            output, index, value)

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_complex(output, index, value):
        cuda.atomic.add(                  # pylint: disable=too-many-function-args
            output.real, index, value.real)
        cuda.atomic.add(                  # pylint: disable=too-many-function-args
            output.imag, index, value.imag)

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_a.shape

        for j in range(col):
            for i in range(row):
                out[j] += in_a[i][j] * in_b[j]

        return out

    def _get_interpolate_cuda():
        """Subroutines for GPU time-domain interpolation (cartesian -> non-cartesian)."""

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate2_cuda(noncart_data, cart_data, interp_value, interp_index):

            # get sizes
            nframes, batch_size, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # gather data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        noncart_data[frame, batch, point] += \
                            val * cart_data[frame, batch, idy, idx]

            return noncart_data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate3_cuda(noncart_data, cart_data, interp_value, interp_index):

            # get sizes
            nframes, batch_size, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # gather data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            noncart_data[frame, batch, point] += \
                                val * cart_data[frame, batch, idz, idy, idx]

            return noncart_data

        return _interpolate2_cuda, _interpolate3_cuda

    def _get_interpolate_lowrank_cuda():
        """
        Subroutines for GPU low-rank interpolation.

        Transform cartesian low rank -> non-cartesian time domain.
        """

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate2_cuda(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

            # get sizes
            ncoeff, batch_size, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # gather data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while gathering data
                        for coeff in range(ncoeff):
                            noncart_data[frame, batch, point] += \
                                val * adjoint_basis[frame, coeff] * \
                                cart_data[coeff, batch, idy, idx]

            return noncart_data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate3_cuda(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

            # get sizes
            ncoeff, batch_size, _, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # gather data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            # do adjoint low rank projection (low-rank subspace -> time domain)
                            # while gathering data
                            for coeff in range(ncoeff):
                                noncart_data[frame, batch, point] += \
                                    val * adjoint_basis[frame, coeff] * \
                                    cart_data[coeff, batch, idz, idy, idx]

            return noncart_data

        return _interpolate2_cuda, _interpolate3_cuda

    def _do_interpolation_cuda2(data_out, data_in, value, index, adjoint_basis):
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)
                         ) // threadsperblock

        data_out = _util.pytorch2numba(data_out)
        data_in = _util.pytorch2numba(data_in)
        value = [_util.pytorch2numba(val) for val in value]
        index = [_util.pytorch2numba(ind) for ind in index]

        # run kernel
        if adjoint_basis is None:
            _get_interpolate_cuda()[0][blockspergrid, threadsperblock](
                data_out, data_in, value, index)
        else:
            adjoint_basis = _util.pytorch2numba(adjoint_basis)
            _get_interpolate_lowrank_cuda()[0][blockspergrid, threadsperblock](
                data_out, data_in, value, index, adjoint_basis)
            adjoint_basis = _util.numba2pytorch(adjoint_basis)

        data_out = _util.numba2pytorch(data_out)
        data_in = _util.numba2pytorch(data_in)
        value = [_util.numba2pytorch(val) for val in value]
        index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]

    def _do_interpolation_cuda3(data_out, data_in, value, index, adjoint_basis):
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)
                         ) // threadsperblock

        data_out = _util.pytorch2numba(data_out)
        data_in = _util.pytorch2numba(data_in)
        value = [_util.pytorch2numba(val) for val in value]
        index = [_util.pytorch2numba(ind) for ind in index]

        # run kernel
        if adjoint_basis is None:
            _get_interpolate_cuda()[1][blockspergrid, threadsperblock](
                data_out, data_in, value, index)
        else:
            adjoint_basis = _util.pytorch2numba(adjoint_basis)
            _get_interpolate_lowrank_cuda()[1][blockspergrid, threadsperblock](
                data_out, data_in, value, index, adjoint_basis)
            adjoint_basis = _util.numba2pytorch(adjoint_basis)

        data_out = _util.numba2pytorch(data_out)
        data_in = _util.numba2pytorch(data_in)
        value = [_util.numba2pytorch(val) for val in value]
        index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]

    do_interpolation_cuda = [_do_interpolation_cuda2, _do_interpolation_cuda3]

    def _get_gridding_cuda(iscomplex):
        """Subroutines for GPU time-domain gridding (non-cartesian -> cartesian)."""
        if iscomplex is True:
            _update = _update_complex
        else:
            _update = _update_real

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _gridding2_cuda(cart_data, noncart_data, interp_value, interp_index):

            # get sizes
            nframes, batch_size, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        _update(cart_data, (frame, batch, idy, idx),
                                val * noncart_data[frame, batch, point])

            return cart_data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _gridding3_cuda(cart_data, noncart_data, interp_value, interp_index):

            # get sizes
            nframes, batch_size, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            _update(cart_data, (frame, batch, idz, idy, idx),
                                    val * noncart_data[frame, batch, point])

            return cart_data

        return _gridding2_cuda, _gridding3_cuda

    def _get_gridding_lowrank_cuda(iscomplex):
        """Subroutines for GPU low-rank gridding (non-cartesian time domain -> cartesian low-rank)."""
        if iscomplex is True:
            _update = _update_complex
        else:
            _update = _update_real

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _gridding2_cuda(cart_data, noncart_data, interp_value, interp_index, basis):

            # get sizes
            ncoeff, batch_size, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while spreading data
                        for coeff in range(ncoeff):
                            _update(cart_data, (coeff, batch, idy, idx),
                                    val * basis[coeff, frame] * noncart_data[frame, batch, point])

            return cart_data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _gridding3_cuda(cart_data, noncart_data, interp_value, interp_index, basis):

            # get sizes
            ncoeff, batch_size, _, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            # do adjoint low rank projection (low-rank subspace -> time domain)
                            # while gathering data
                            for coeff in range(ncoeff):
                                _update(cart_data, (coeff, batch, idz, idy, idx),
                                        val * basis[coeff, frame] * noncart_data[frame, batch, point])

            return cart_data

        return _gridding2_cuda, _gridding3_cuda

    def _do_gridding_cuda2(data_out, data_in, value, index, basis):
        is_complex = torch.is_complex(data_in)  # pylint: disable=no-member

        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)
                         ) // threadsperblock

        data_out = _util.pytorch2numba(data_out)
        data_in = _util.pytorch2numba(data_in)
        value = [_util.pytorch2numba(val) for val in value]
        index = [_util.pytorch2numba(ind) for ind in index]

        # run kernel
        if basis is None:
            _get_gridding_cuda(is_complex)[0][blockspergrid, threadsperblock](
                data_out, data_in, value, index)
        else:
            basis = _util.pytorch2numba(basis)
            _get_gridding_lowrank_cuda(is_complex)[0][blockspergrid, threadsperblock](
                data_out, data_in, value, index, basis)
            basis = _util.numba2pytorch(basis)

        data_out = _util.numba2pytorch(data_out)
        data_in = _util.numba2pytorch(data_in)
        value = [_util.numba2pytorch(val) for val in value]
        index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]

    def _do_gridding_cuda3(data_out, data_in, value, index, basis):
        is_complex = torch.is_complex(data_in)  # pylint: disable=no-member

        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)
                         ) // threadsperblock

        data_out = _util.pytorch2numba(data_out)
        data_in = _util.pytorch2numba(data_in)
        value = [_util.pytorch2numba(val) for val in value]
        index = [_util.pytorch2numba(ind) for ind in index]

        # run kernel
        if basis is None:
            _get_gridding_cuda(is_complex)[1][blockspergrid, threadsperblock](
                data_out, data_in, value, index)
        else:
            basis = _util.pytorch2numba(basis)
            _get_gridding_lowrank_cuda(is_complex)[1][blockspergrid, threadsperblock](
                data_out, data_in, value, index, basis)
            basis = _util.numba2pytorch(basis)

        data_out = _util.numba2pytorch(data_out)
        data_in = _util.numba2pytorch(data_in)
        value = [_util.numba2pytorch(val) for val in value]
        index = [_util.numba2pytorch(ind, requires_grad=False) for ind in index]

    do_gridding_cuda = [_do_gridding_cuda2, _do_gridding_cuda3]

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_selfadjoint_cuda(data_out, toeplitz_matrix, data_in):

        n_coeff, batch_size, _ = data_in.shape

        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < n_coeff*batch_size:
            coeff = i // batch_size
            batch = i % batch_size

            _dot_product_cuda(
                data_out[coeff][batch], toeplitz_matrix[coeff], data_in[coeff][batch])

        return data_out

    def do_selfadjoint_interpolation_cuda(data_out, data_in, toeplitz_matrix):
        """ wrapper for CUDA self-adjoint. """
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)
                         ) // threadsperblock

        data_out = _util.pytorch2numba(data_out)
        data_in = _util.pytorch2numba(data_in)
        toeplitz_matrix = _util.pytorch2numba(toeplitz_matrix)

        # run kernel
        _interp_selfadjoint_cuda[blockspergrid, threadsperblock](
            data_out, toeplitz_matrix, data_in)

        data_out = _util.numba2pytorch(data_out)
        data_in = _util.numba2pytorch(data_in)
        toeplitz_matrix = _util.numba2pytorch(toeplitz_matrix)
