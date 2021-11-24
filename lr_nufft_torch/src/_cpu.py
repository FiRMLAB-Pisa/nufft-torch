# -*- coding: utf-8 -*-
"""
CPU specific functions.

@author: Matteo
"""
# pylint: disable=not-an-iterable
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=protected-access

from typing import Tuple, Callable

import numpy as np
import numba as nb

from lr_nufft_torch.src import _common


class _DeGridding:

    apply: Callable

    def __init__(self, kernel_dict, basis_adjoint=None):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_dict['sparse_coefficients']
        kernel_width = kernel_dict['width']

        # get kernel neighbourhood
        kernel_neighbourhood = _iterator._get_neighbourhood(kernel_width)

        # select correct sub-routine
        if basis_adjoint is None:
            callback = _Interpolation._get_callback()

            def _apply(noncart_data, cart_data):
                return callback(noncart_data, cart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood)

        else:
            callback = _Interpolation._get_lowrank_callback()

            def _apply(noncart_data, cart_data):
                return callback(noncart_data, cart_data,
                                kernel_sparse_coefficients,
                                kernel_neighbourhood,
                                basis_adjoint)

        # assign
        self.apply = _apply

    @staticmethod
    def _get_callback():

        # kernel function
        kernel = _kernel._evaluate

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes*batch_size*npts):

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

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis_adjoint):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis_adjoint.shape[-1]

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes*batch_size*npts):

                # get current frame and k-space index
                frame, batch, target = _get_target_point(
                    i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(
                        target, point, kernel_sparse_coefficients)

                    # update
                    gather(noncart_data, cart_data,
                           (batch, frame, target, source), value,
                           basis_adjoint, ncoeff)

        return _callback


class _Gridding:

    apply: Callable

    def __init__(self, kernel_dict, basis=None, sharing_width=None):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_dict['sparse_coefficients']
        kernel_width = kernel_dict['width']

        # get kernel neighbourhood
        kernel_neighbourhood = _iterator._get_neighbourhood(kernel_width)

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
        self.apply = _apply

    @staticmethod
    def _get_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_batch_and_frame

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames and batches
            for i in nb.prange(nframes*batch_size):

                # get current frame and k-space index
                frame, batch = _get_source_point(i, batch_size)

                # iterate over readout points
                for source in range(npts):

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
        _get_source_point = _iterator._get_noncart_points_parallelize_over_batch_and_frame

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_viewshare

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      sharing_width):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # parallelize over frames and batches
            for i in nb.prange(nframes*batch_size):

                # get current frame and k-space index
                frame, batch = _get_source_point(i, batch_size)

                # iterate over readout points
                for source in range(npts):

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
        _get_source_point = _iterator._get_noncart_points_parallelize_over_batch_and_frame

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff*batch_size):

                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(nframes):

                    # iterate over readout points
                    for source in range(npts):

                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                source, point, kernel_sparse_coefficients)

                            # update
                            spread(cart_data, noncart_data,
                                   (batch, coeff, target, frame, source),
                                   value, basis)

        return _callback

    @staticmethod
    def _get_viewshare_lowrank_callback():

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_batch_and_frame

        # kernel function
        kernel = _kernel._evaluate

        # spread function
        spread = _spread._data_viewshare_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      sharing_width,
                      basis):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff*batch_size):

                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(nframes):

                    # iterate over readout points
                    for source in range(npts):

                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                source, point, kernel_sparse_coefficients)

                            # update
                            spread(cart_data, noncart_data,
                                   (batch, coeff, target, frame, source),
                                   value, sharing_width, nframes, basis)

        return _callback


class _iterator(_common._iterator):

    _get_noncart_points_parallelize_over_all = staticmethod(nb.njit(
        _common._iterator._get_noncart_points_parallelize_over_all,
        fastmath=True, cache=True))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_noncart_points_parallelize_over_batch_and_frame(index: int,
                                                             batch_size: int) -> Tuple[int, int]:

        frame = index // batch_size
        batch = index % batch_size

        return frame, batch

    _check_boundaries = staticmethod(nb.njit(
        _common._iterator._check_boundaries, fastmath=True, cache=True))


class _kernel(_common._kernel):

    # Main Kaiser-Bessel kernel function
    @staticmethod
    @nb.njit(fastmath=True, cache=True)  # pragma: no cover
    def _function(x_0, beta):
        if abs(x_0) > 1:
            value = 0

        x_0 = beta * (1 - x_0**2)**0.5
        x_i = x_0 / 3.75
        if x_i < 3.75:
            value = 1 + 3.5156229 * x_i**2 + 3.0899424 * x_i**4 +\
                1.2067492 * x_i**6 + 0.2659732 * x_i**8 +\
                0.0360768 * x_i**10 + 0.0045813 * x_i**12
        else:
            value = x_0**-0.5 * np.exp(x_0) * (
                0.39894228 + 0.01328592 * x_i**-1 +
                0.00225319 * x_i**-2 - 0.00157565 * x_i**-3 +
                0.00916281 * x_i**-4 - 0.02057706 * x_i**-5 +
                0.02635537 * x_i**-6 - 0.01647633 * x_i**-7 +
                0.00392377 * x_i**-8)

        return value

    # Utililities
    _prod = staticmethod(
        nb.njit(_common._kernel._prod, fastmath=True, cache=True))

    _ravel_index = staticmethod(
        nb.njit(_common._kernel._ravel_index, fastmath=True, cache=True))

    # precomputed Kernel evaluation
    _evaluate = staticmethod(
        nb.njit(_common._kernel._evaluate, fastmath=True, cache=True))


class _gather(_common._gather):

    _data = staticmethod(
        nb.njit(_common._gather._data, fastmath=True, cache=True))

    _data_lowrank = staticmethod(
        nb.njit(_common._gather._data_lowrank, fastmath=True, cache=True))


class _spread:

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data(data_out, data_in, index_tuple, kernel_value):

        # unpack indexes
        batch, frame, index_out, index_in = index_tuple

        # get input and output locations
        idx_out = (frame, batch, index_out)
        idx_in = (frame, batch, index_in)

        # update data point
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
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
            data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data_lowrank(data_out, data_in, index_tuple, kernel_value,
                      basis):

        # unpack indexes
        batch, coeff, index_out, frame, index_in = index_tuple

        # get output and input locations
        idx_out = (coeff, batch, index_out)
        idx_in = (frame, batch, index_in)

        # get total weight
        weight = kernel_value * basis[coeff, frame]

        # update data point
        data_out[idx_out] += weight * data_in[idx_in]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data_viewshare_lowrank(data_out, data_in, index_tuple, kernel_value,
                                share_width, nframes, basis):

        # unpack indexes
        batch, coeff, index_out, frame, index_in = index_tuple

        # get output and input locations
        idx_out = (coeff, batch, index_out)
        idx_in = (frame, batch, index_in)

        # iterate over frames within sharing window
        for dframe in range(-share_width // 2, share_width // 2):
            # get input frame
            frame_in = _iterator._check_boundaries(frame + dframe, nframes)

            # get total weight
            weight = kernel_value * basis[coeff, frame_in]

            # update data point
            data_out[idx_out] += weight * data_in[idx_in]
