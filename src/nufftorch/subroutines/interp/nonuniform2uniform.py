"""Routines to perform gridding of (non-Cartesian) under-sampled data."""
import numba as nb
import torch


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_cpu_1d(cart_data, noncart_data, interp_value, interp_index):
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = interp_index
    xvalue = interp_value

    # get interpolator width
    xwidth = xindex.shape[-1]

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            # spread data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                cart_data[frame, batch, idx] += val * noncart_data[frame, batch, point]

    return cart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_cpu_2d(cart_data, noncart_data, interp_value, interp_index):
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
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
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

                    cart_data[frame, batch, idy, idx] += (
                        val * noncart_data[frame, batch, point]
                    )

    return cart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_cpu_3d(cart_data, noncart_data, interp_value, interp_index):
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
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
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

                        cart_data[frame, batch, idz, idy, idx] += (
                            val * noncart_data[frame, batch, point]
                        )

    return cart_data


if torch.cuda.is_available():

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_1d(cart_data, noncart_data, interp_value, interp_index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index
        xvalue = interp_value

        # get interpolator width
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                nb.cuda.atomic.add(
                    cart_data,
                    (frame, batch, idx),
                    val * noncart_data[frame, batch, point],
                )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_2d(cart_data, noncart_data, interp_value, interp_index):
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
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    nb.cuda.atomic.add(
                        cart_data,
                        (frame, batch, idy, idx),
                        val * noncart_data[frame, batch, point],
                    )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_3d(cart_data, noncart_data, interp_value, interp_index):
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
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
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

                        nb.cuda.atomic.add(
                            cart_data,
                            (frame, batch, idz, idy, idx),
                            val * noncart_data[frame, batch, point],
                        )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_cplx_1d(
        cart_data, noncart_data, interp_value, interp_index
    ):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index
        xvalue = interp_value

        # get interpolator width
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                nb.cuda.atomic.add(
                    cart_data.real,
                    (frame, batch, idx),
                    (val * noncart_data[frame, batch, point]).real,
                )
                nb.cuda.atomic.add(
                    cart_data.imag,
                    (frame, batch, idx),
                    (val * noncart_data[frame, batch, point]).imag,
                )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_cplx_2d(
        cart_data, noncart_data, interp_value, interp_index
    ):
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
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    nb.cuda.atomic.add(
                        cart_data.real,
                        (frame, batch, idy, idx),
                        (val * noncart_data[frame, batch, point]).real,
                    )
                    nb.cuda.atomic.add(
                        cart_data.imag,
                        (frame, batch, idy, idx),
                        (val * noncart_data[frame, batch, point]).imag,
                    )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_gpu_cplx_3d(
        cart_data, noncart_data, interp_value, interp_index
    ):
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
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
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

                        nb.cuda.atomic.add(
                            cart_data.real,
                            (frame, batch, idz, idy, idx),
                            (val * noncart_data[frame, batch, point]).real,
                        )
                        nb.cuda.atomic.add(
                            cart_data.imag,
                            (frame, batch, idz, idy, idx),
                            (val * noncart_data[frame, batch, point]).imag,
                        )

        return cart_data
