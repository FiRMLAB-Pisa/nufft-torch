# -*- coding: utf-8 -*-
"""
CUDA specific subroutines.

@author: Matteo Cencini
"""
# pylint: disable=not-an-iterable
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=protected-access
# pylint: disable=too-many-function-args
# pylint: disable=unsubscriptable-object
# pylint: disable=arguments-differ

from typing import Callable

from numba import cuda

from nufftorch.src import _common


_dot_product = cuda.jit(_common._dot_product, device=True, inline=True)


@cuda.jit(fastmath=True)  # pragma: no cover
def _batched_dot_product(data_out, data_in, matrix):

    n_coeff, batch_size, _ = data_in.shape

    i = cuda.grid(1)
    if i < n_coeff*batch_size:
        coeff = i // batch_size
        batch = i % batch_size

        _dot_product(data_out[coeff][batch],
                     matrix[coeff], data_in[coeff][batch])

    return data_out


class _DeGridding:

    apply: Callable

    def __init__(self, data_size, kernel_dict, basis_adjoint, threadsperblock):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_dict['sparse_coefficients']
        kernel_width = kernel_dict['width']

        # get kernel neighbourhood
        kernel_neighbourhood = _iterator._get_neighbourhood(kernel_width)

        # calculate blocks per grid
        blockspergrid = int((data_size + (threadsperblock - 1)) // threadsperblock)

        # select correct sub-routine
        if basis_adjoint is None:
            callback = _DeGridding._get_callback()

            def _apply(noncart_data, cart_data):
                return callback(noncart_data, cart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood)

        else:
            callback = _DeGridding._get_lowrank_callback()

            def _apply(noncart_data, cart_data):
                return callback(noncart_data, cart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood,
                                basis_adjoint)

        # assign
        self.apply = _apply[blockspergrid, threadsperblock]

    @staticmethod
    def _get_callback():

        # kernel function
        kernel = _kernel._evaluate

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, target = _get_target_point(
                    i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(
                        target, point, kernel_sparse_coefficients)

                    # update
                    gather(noncart_data, cart_data,
                           (batch, frame, target, source), value)

        return _callback

    @staticmethod
    def _get_lowrank_callback():

        # kernel function
        kernel = _kernel._evaluate

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data_lowrank

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis_adjoint):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, target = _get_target_point(
                    i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(
                        target, point, kernel_sparse_coefficients)

                    # update
                    gather(noncart_data, cart_data,
                           (batch, frame, target, source), value, basis_adjoint)

        return _callback


class _Gridding:

    apply: Callable

    def __init__(self, data_size, kernel_dict, basis, sharing_width, threadsperblock):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_dict['sparse_coefficients']
        kernel_width = kernel_dict['width']

        # get kernel neighbourhood
        kernel_neighbourhood = _iterator._get_neighbourhood(kernel_width)

        # calculate blocks per grid
        blockspergrid = int((data_size + (threadsperblock - 1)) // threadsperblock)

        # select correct sub-routine
        if basis is None and sharing_width is None:
            callback = _Gridding._get_callback()

            def _apply(cart_data, noncart_data):
                return callback(cart_data, noncart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood)
        elif basis is None and sharing_width is not None:
            callback = _Gridding._get_viewshare_callback()

            def _apply(cart_data, noncart_data):
                return callback(cart_data, noncart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood,
                                basis)
        elif basis is not None and sharing_width is None:
            callback = _Gridding._get_lowrank_callback()

            def _apply(cart_data, noncart_data):
                return callback(cart_data, noncart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood,
                                sharing_width)
        else:
            callback = _Gridding._get_viewshare_lowrank_callback()

            def _apply(cart_data, noncart_data):
                return callback(cart_data, noncart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood,
                                sharing_width,
                                basis)

        # assign
        self.apply = _apply[blockspergrid, threadsperblock]

    @staticmethod
    def _get_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(
                    i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(
                        source, point, kernel_sparse_coefficients)

                    # update
                    spread(cart_data, noncart_data,
                           (batch, frame, target, source),
                           value)

        return _callback

    @staticmethod
    def _get_viewshare_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_viewshare

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      sharing_width):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(
                    i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(
                        source, point, kernel_sparse_coefficients)

                    # update
                    spread(cart_data, noncart_data,
                           (batch, frame, target, source),
                           value, sharing_width, nframes)

        return _callback

    @staticmethod
    def _get_lowrank_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_lowrank

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(
                    i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(
                        source, point, kernel_sparse_coefficients)

                    # update
                    spread(cart_data, noncart_data,
                           (batch, frame, target, source),
                           value, basis, ncoeff)

        return _callback

    @staticmethod
    def _get_viewshare_lowrank_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_viewshare_lowrank

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      sharing_width,
                      basis):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(
                    i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(
                        source, point, kernel_sparse_coefficients)

                    # update
                    spread(cart_data, noncart_data,
                           (batch, frame, target, source),
                           value, sharing_width, nframes, basis, ncoeff)

        return _callback


class _iterator(_common._iterator):

    _get_noncart_points_parallelize_over_all = staticmethod(cuda.jit(
        _common._iterator._get_noncart_points_parallelize_over_all,
        device=True, inline=True))

    _check_boundaries = staticmethod(cuda.jit(
        _common._iterator._check_boundaries, device=True, inline=True))


class _kernel(_common._kernel):

    # Utililities
    _prod = staticmethod(
        cuda.jit(_common._kernel._prod, device=True, inline=True))

    _ravel_index = staticmethod(
        cuda.jit(_common._kernel._ravel_index, device=True, inline=True))

    # precomputed Kernel evaluation
    _evaluate = staticmethod(
        cuda.jit(_common._kernel._evaluate, device=True, inline=True))


class _gather(_common._gather):

    _data = staticmethod(
        cuda.jit(_common._gather._data, device=True, inline=True))

    _data_lowrank = staticmethod(
        cuda.jit(_common._gather._data_lowrank, device=True, inline=True))


class _spread:

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update(output, index, value):
        cuda.atomic.add(
            output.real, index, value.real)
        cuda.atomic.add(
            output.imag, index, value.imag)

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data(data_out, data_in, index_tuple, kernel_value):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # get input and output locations
        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)

        # update data point
        data_out = _spread._update(
            data_out, idx_out, kernel_value * data_in[idx_in])

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data_viewshare(data_out, data_in, index_tuple, kernel_value,
                        share_width, nframes):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # get output location
        idx_out = (frame, batch, index_out)

        # iterate over frames within sharing window
        for dframe in range(-share_width // 2, share_width // 2):
            # get input frame
            frame_in = _iterator._check_boundaries(frame + dframe, nframes)

            # get input location
            idx_in = (frame_in, batch, index_in)

            # update data point
            data_out = _spread._update(
                data_out, idx_out, kernel_value * data_in[idx_in])

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data_lowrank(data_out, data_in, index_tuple, kernel_value,
                      basis, ncoeff):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # get input locations
        idx_in = (frame, batch, index_in)

        # iterate over low rank coefficients
        for coeff in range(ncoeff):
            # get output frame
            idx_out = (coeff, batch, index_out)

            # get total weight
            weight = kernel_value * basis[coeff, frame]

            # update data point
            data_out = _spread._update(
                data_out, idx_out, weight * data_in[idx_in])

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data_viewshare_lowrank(data_out, data_in, index_tuple, kernel_value,
                                share_width, nframes, basis, ncoeff):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # iterate over low rank coefficients
        for coeff in range(ncoeff):
            # get output frame
            idx_out = (coeff, batch, index_out)

            # iterate over frames within sharing window
            for dframe in range(-share_width // 2, share_width // 2):
                # get input frame
                frame_in = _iterator._check_boundaries(frame + dframe, nframes)

                # get input location
                idx_in = (frame_in, batch, index_in)

                # get total weight
                weight = kernel_value * basis[coeff, frame_in]

                # update data point
                data_out = _spread._update(
                    data_out, idx_out, weight * data_in[idx_in])
                