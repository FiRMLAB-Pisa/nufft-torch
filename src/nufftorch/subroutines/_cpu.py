"""CPU specific subroutines."""
__all__ = []

from typing import Tuple, Callable

import numpy as np
import numba as nb

from nufftorch.subroutines import _common


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def prepare_sparse_coefficient_matrix(value, index, coord, beta, shape):
    """
    Prepare interpolator matrix for KB gridding / inverse-gridding along a single (1D) axis.

    Args:
        value (np.ndarray): output matrix storing (float) interpolation coefficients.
        index (np.ndarray): output matrix storing (int) uniform coordinates for interpolated samples.
        coord (np.ndarray): input (non-uniform) sample coordinates.
        beta (float): KB-parameter.
        shape (np.ndarray): output (1D) grid shape.

    """
    # get sizes
    npts = coord.shape[0]
    width = index.shape[-1]

    for i in nb.prange(npts):
        x_0 = np.ceil(coord[i] - width / 2)

        for x_i in range(width):
            val = _kernel_function(((x_0 + x_i) - coord[i]) / (width / 2), beta)

            # save interpolator
            value[i, x_i] = val
            index[i, x_i] = (x_0 + x_i) % shape


_dot_product = nb.njit(_common._dot_product, fastmath=True, cache=True)


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def batched_dot_product(data_out, data_in, matrix):
    """
    Perform batched dot product between a stacks of vectors and a matrix.

    Args:
        data_out (np.ndarray): output stack of vectors of shape of shape (nbatches, ncols).
        data_in (np.ndarray): stack of flattened matrices of shape (nbatches, nrows).
        matrix (np.ndarray): flattened matrix of shape (nrows, ncols).

    """
    n_points, batch_size, _ = data_in.shape

    for i in nb.prange(n_points * batch_size):
        point = i // batch_size
        batch = i % batch_size
        _dot_product(data_out[point][batch], data_in[point][batch], matrix[point])


class InverseGriddingNN:
    """Inverse (uniform-to-nonuniform) Nearest-Neighbour (NN) interpolation."""

    apply: Callable

    def __init__(self, data_size, index_tuple, basis_adjoint, threadsperblock):
        # infer shape
        ndim = index_tuple[0].shape[0]

        # select correct sub-routine
        if basis_adjoint is None:
            callback = InverseGriddingNN._get_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback(noncart_data, cart_data, index_tuple)

        else:
            callback = InverseGriddingNN._get_lowrank_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback(noncart_data, cart_data, index_tuple, basis_adjoint)

        # assign
        InverseGriddingNN.__call__ = _apply

    @staticmethod
    def _get_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(noncart_data, cart_data, index_tuple):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack index tuple: index and width (x, y, z) + grid shape (nx, ny, nz)
            idx, _, grid_off = index_tuple

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes * batch_size * npts):
                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data from point
                source = kernel(frame, target, idx, grid_off)

                # update
                gather(noncart_data, cart_data, frame, batch, target, source)

        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(noncart_data, cart_data, index_tuple, basis_adjoint):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis_adjoint.shape[-1]

            # unpack index tuple: index and width (x, y, z) + grid shape (nx, ny, nz)
            idx, _, grid_off = index_tuple

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes * batch_size * npts):
                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data from point
                source = kernel(frame, target, idx, grid_off)

                # update
                gather(
                    noncart_data, cart_data, frame, batch, source, basis_adjoint, ncoeff
                )

        return _callback


class GriddingNN:
    """Forward (nonuniform-to-uniform) Nearest-Neighbour (NN) interpolation."""

    apply: Callable

    def __init__(self, data_size, kernel_tuple, basis, threadsperblock):
        # unpack kernel dict
        kernel_sparse_coefficients = kernel_tuple
        kernel_width = kernel_tuple[-2]
        ndim = kernel_tuple[0].shape[0]

        # get kernel neighbourhood
        kernel_neighbourhood = np.array(_iterator._get_neighbourhood(*kernel_width))

        # select correct sub-routine
        if basis is None:
            callback = GriddingKB._get_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback(
                    cart_data,
                    noncart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                )

        else:
            callback = GriddingKB._get_lowrank_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback(
                    cart_data,
                    noncart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                    basis,
                )

        # assign
        GriddingKB.__call__ = _apply

    @staticmethod
    def _get_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data, noncart_data, kernel_sparse_coefficients, kernel_neighbourhood
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames and batches
            for i in nb.prange(nframes * batch_size):
                # get current frame and k-space index
                frame, batch = _get_source_point(i, batch_size)

                # iterate over readout points
                for source in range(npts):
                    # spread data within kernel radius
                    for point in kernel_neighbourhood:
                        value, target = kernel(
                            frame, source, point, kvalue, kidx, grid_off
                        )

                        # update
                        spread(
                            cart_data, noncart_data, frame, batch, source, target, value
                        )

        return _callback

    @staticmethod
    def _get_viewsharing_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame + share_idx,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_swkh_viewsharing_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
            spatial_weight,
            temporal_weight,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update kernel according to spatial weight
                            value *= spatial_weight[target]

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # select temporal weight
                                tw = temporal_weight[frame, hwidth]
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    tw * value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(nframes):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update
                            spread(
                                cart_data,
                                noncart_data,
                                frame,
                                batch,
                                source,
                                coeff,
                                target,
                                value,
                                basis,
                            )

        return _callback

    @staticmethod
    def _get_viewsharing_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame + share_idx,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_swkh_viewsharing_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("NN")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
            spatial_weight,
            temporal_weight,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update kernel according to spatial weight
                            value *= spatial_weight[target]

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # select temporal weight
                                tw = temporal_weight[frame, hwidth]
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    tw * value,
                                    basis,
                                )

        return _callback


class InverseGriddingKB:
    """Inverse (uniform-to-nonuniform) Kaiser-Bessel (KB) interpolation."""

    apply: Callable

    def __init__(self, data_size, kernel_tuple, basis_adjoint, threadsperblock):
        # unpack kernel dict
        kernel_sparse_coefficients = kernel_tuple
        kernel_width = kernel_tuple[-2]
        ndim = kernel_tuple[0].shape[0]

        # get kernel neighbourhood
        kernel_neighbourhood = np.array(_iterator._get_neighbourhood(*kernel_width))

        # select correct sub-routine
        if basis_adjoint is None:
            callback = InverseGriddingKB._get_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback(
                    noncart_data,
                    cart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                )

        else:
            callback = InverseGriddingKB._get_lowrank_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback(
                    noncart_data,
                    cart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                    basis_adjoint,
                )

        # assign
        InverseGriddingKB.__call__ = _apply

    @staticmethod
    def _get_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            noncart_data, cart_data, kernel_sparse_coefficients, kernel_neighbourhood
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes * batch_size * npts):
                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(frame, target, point, kvalue, kidx, grid_off)

                    # update
                    gather(noncart_data, cart_data, frame, batch, target, source, value)

        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            noncart_data,
            cart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis_adjoint,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis_adjoint.shape[-1]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            for i in nb.prange(nframes * batch_size * npts):
                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(frame, target, point, kvalue, kidx, grid_off)

                    # update
                    gather(
                        noncart_data,
                        cart_data,
                        frame,
                        batch,
                        target,
                        source,
                        value,
                        basis_adjoint,
                        ncoeff,
                    )

        return _callback


class GriddingKB:
    """Forward (nonuniform-to-uniform) Kaiser-Bessel (KB) interpolation."""

    apply: Callable

    def __init__(self, data_size, kernel_tuple, basis, threadsperblock):
        # unpack kernel dict
        kernel_sparse_coefficients = kernel_tuple
        kernel_width = kernel_tuple[-2]
        ndim = kernel_tuple[0].shape[0]

        # get kernel neighbourhood
        kernel_neighbourhood = np.array(_iterator._get_neighbourhood(*kernel_width))

        # select correct sub-routine
        if basis is None:
            callback = GriddingKB._get_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback(
                    cart_data,
                    noncart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                )

        else:
            callback = GriddingKB._get_lowrank_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback(
                    cart_data,
                    noncart_data,
                    kernel_sparse_coefficients,
                    kernel_neighbourhood,
                    basis,
                )

        # assign
        GriddingKB.__call__ = _apply

    @staticmethod
    def _get_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data, noncart_data, kernel_sparse_coefficients, kernel_neighbourhood
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames and batches
            for i in nb.prange(nframes * batch_size):
                # get current frame and k-space index
                frame, batch = _get_source_point(i, batch_size)

                # iterate over readout points
                for source in range(npts):
                    # spread data within kernel radius
                    for point in kernel_neighbourhood:
                        value, target = kernel(
                            frame, source, point, kvalue, kidx, grid_off
                        )

                        # update
                        spread(
                            cart_data, noncart_data, frame, batch, source, target, value
                        )

        return _callback

    @staticmethod
    def _get_viewsharing_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame + share_idx,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_swkh_viewsharing_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
            spatial_weight,
            temporal_weight,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update kernel according to spatial weight
                            value *= spatial_weight[target]

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # select temporal weight
                                tw = temporal_weight[frame, hwidth]
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    tw * value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(nframes):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update
                            spread(
                                cart_data,
                                noncart_data,
                                frame,
                                batch,
                                source,
                                coeff,
                                target,
                                value,
                                basis,
                            )

        return _callback

    @staticmethod
    def _get_viewsharing_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame + share_idx,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    value,
                                    basis,
                                )

        return _callback

    @staticmethod
    def _get_swkh_viewsharing_lowrank_callback(ndim):
        # kernel function
        kernel = _kernel._evaluate("KB")[ndim - 1]

        # iterator function
        _get_source_point = (
            _iterator._get_noncart_points_parallelize_over_batch_and_frame
        )

        # spread function
        spread = _spread._data_lowrank

        @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
        def _callback(
            cart_data,
            noncart_data,
            kernel_sparse_coefficients,
            kernel_neighbourhood,
            basis,
            window_width,
            spatial_weight,
            temporal_weight,
        ):
            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # get half window width
            hwidth = window_width // 2

            # parallelize over frames, batches and k-space points
            for i in nb.prange(ncoeff * batch_size):
                # get current frame and k-space index
                coeff, batch = _get_source_point(i, batch_size)

                # iterate over frames
                for frame in range(hwidth, nframes - hwidth):
                    # iterate over readout points
                    for source in range(npts):
                        # spread data within kernel radius
                        for point in kernel_neighbourhood:
                            value, target = kernel(
                                frame, source, point, kvalue, kidx, grid_off
                            )

                            # update kernel according to spatial weight
                            value *= spatial_weight[target]

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # select temporal weight
                                tw = temporal_weight[frame, hwidth]
                                # update target
                                spread(
                                    cart_data,
                                    noncart_data,
                                    frame,
                                    batch,
                                    source,
                                    coeff,
                                    target,
                                    tw * value,
                                    basis,
                                )

        return _callback


# %% local utils
class _iterator(_common._iterator):
    _get_noncart_points_parallelize_over_all = staticmethod(
        nb.njit(
            _common._iterator._get_noncart_points_parallelize_over_all,
            fastmath=True,
            cache=True,
        )
    )

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_noncart_points_parallelize_over_batch_and_frame(
        index: int, batch_size: int
    ) -> Tuple[int, int]:
        frame = index // batch_size
        batch = index % batch_size

        return frame, batch


class _kernel(_common._kernel):
    # Main Kaiser-Bessel kernel function
    @staticmethod
    @nb.njit(fastmath=True, cache=True)  # pragma: no cover
    def _function(x_0, beta):
        if abs(x_0) > 1:
            value = 0

        x_0 = beta * (1 - x_0**2) ** 0.5
        x_i = x_0 / 3.75
        if x_i < 3.75:
            value = (
                1
                + 3.5156229 * x_i**2
                + 3.0899424 * x_i**4
                + 1.2067492 * x_i**6
                + 0.2659732 * x_i**8
                + 0.0360768 * x_i**10
                + 0.0045813 * x_i**12
            )
        else:
            value = (
                x_0**-0.5
                * np.exp(x_0)
                * (
                    0.39894228
                    + 0.01328592 * x_i**-1
                    + 0.00225319 * x_i**-2
                    - 0.00157565 * x_i**-3
                    + 0.00916281 * x_i**-4
                    - 0.02057706 * x_i**-5
                    + 0.02635537 * x_i**-6
                    - 0.01647633 * x_i**-7
                    + 0.00392377 * x_i**-8
                )
            )

        return value

    # precomputed Kernel evaluation
    @staticmethod
    def _evaluate():
        # get base functions
        _evaluate_1d, _evaluate_2d, _evaluate_3d = _common._kernel._make_evaluate()

        # jit
        _evaluate_1d = nb.njit(_evaluate_1d, fastmath=True, cache=True)
        _evaluate_2d = nb.njit(_evaluate_2d, fastmath=True, cache=True)
        _evaluate_3d = nb.njit(_evaluate_3d, fastmath=True, cache=True)

        return _evaluate_1d, _evaluate_2d, _evaluate_3d


_kernel_function = _kernel._function


class _gather(_common._gather):
    _data = staticmethod(nb.njit(_common._gather._data, fastmath=True, cache=True))

    _data_lowrank = staticmethod(
        nb.njit(_common._gather._data_lowrank, fastmath=True, cache=True)
    )


class _spread:
    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data(data_out, data_in, frame, batch, index_in, index_out, kernel_value):
        # get input and output locations
        idx_in = (frame, batch, index_in)
        idx_out = (frame, batch, index_out)

        # update data point
        data_out[idx_out] += kernel_value * data_in[idx_in]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _data_lowrank(
        data_out, data_in, frame, batch, index_in, coeff, index_out, kernel_value, basis
    ):
        # get output and input locations
        idx_in = (frame, batch, index_in)
        idx_out = (coeff, batch, index_out)

        # get total weight
        weight = kernel_value * basis[coeff, frame]

        # update data point
        data_out[idx_out] += weight * data_in[idx_in]
