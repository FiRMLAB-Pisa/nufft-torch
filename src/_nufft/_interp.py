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
# from typing import List, Tuple, Union

import numpy as np
import numba as nb

import torch
# from torch import Tensor

#%% CPU specific routines


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
            batch = tmp // batch_size
            point = tmp % batch_size

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
            batch = tmp // batch_size
            point = tmp % batch_size

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
    def _interpolate2(noncart_data, cart_data, interp_value, interp_index, basis_adj):

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
            batch = tmp // batch_size
            point = tmp % batch_size

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
                            val * basis_adj[frame, coeff] * \
                            cart_data[coeff, batch, idy, idx]

        return noncart_data

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _interpolate3(noncart_data, cart_data, interp_value, interp_index, basis_adj):

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
            batch = tmp // batch_size
            point = tmp % batch_size

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
                                val * basis_adj[frame, coeff] * \
                                cart_data[coeff, batch, idz, idy, idx]

        return noncart_data

    return _interpolate2, _interpolate3


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


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp_selfadjoint(data_out, toeplitz_matrix, data_in):

    n_coeff, batch_size, _ = data_in.shape

    for i in nb.prange(n_coeff * batch_size): # pylint: disable=not-an-iterable
        coeff = i // batch_size
        batch = i % batch_size

        _dot_product(data_out[coeff][batch],
                     toeplitz_matrix[coeff], data_in[coeff][batch])

    return data_out


# %% GPU specific routines
if torch.cuda.is_available():
    # CUDA settings
    threadsperblock = 32

    @nb.cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_real(output, index, value):
        nb.cuda.atomic.add(        # pylint: disable=too-many-function-args
            output, index, value)

    @nb.cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_complex(output, index, value):
        nb.cuda.atomic.add(                  # pylint: disable=too-many-function-args
            output.real, index, value.real)
        nb.cuda.atomic.add(                  # pylint: disable=too-many-function-args
            output.imag, index, value.imag)

    @nb.cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_a.shape

        for j in range(col):
            for i in range(row):
                out[j] += in_a[i][j] * in_b[j]

        return out

    def _get_interpolate_cuda():
        """Subroutines for GPU time-domain interpolation (cartesian -> non-cartesian)."""

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate2_cuda(noncart_data, cart_data, interp_value, interp_index, basis_adj):

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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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
                                val * basis_adj[frame, coeff] * \
                                cart_data[coeff, batch, idy, idx]

            return noncart_data

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
        def _interpolate3_cuda(noncart_data, cart_data, interp_value, interp_index, basis_adj):

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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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
                                    val * basis_adj[frame, coeff] * \
                                    cart_data[coeff, batch, idz, idy, idx]

            return noncart_data

        return _interpolate2_cuda, _interpolate3_cuda

    def _get_gridding_cuda(iscomplex):
        """Subroutines for GPU time-domain gridding (non-cartesian -> cartesian)."""
        if iscomplex is True:
            _update = _update_complex
        else:
            _update = _update_real

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

        @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
            i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame = i // (batch_size*npts)
                tmp = i % (batch_size*npts)
                batch = tmp // batch_size
                point = tmp % batch_size

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

    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_selfadjoint_cuda(data_out, toeplitz_matrix, data_in):

        n_coeff, batch_size, _ = data_in.shape

        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < n_coeff*batch_size:
            coeff = i // batch_size
            batch = i % batch_size

            _dot_product_cuda(
                data_out[coeff][batch], toeplitz_matrix[coeff], data_in[coeff][batch])

        return data_out
