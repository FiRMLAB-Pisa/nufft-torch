import numba as nb
import torch


def uniform2nonuniform():
    pass


# %% CPU
@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _uniform2nonuniform_cpu_1d(noncart_data, cart_data, interp_value, interp_index):
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = interp_index
    xvalue = interp_value

    # get interpolator width
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_x in range(xwidth):
            idx = xindex[frame, point, i_x]
            val = xvalue[frame, point, i_x]

            noncart_data[frame, batch, point] += val * cart_data[frame, batch, idx]

    return noncart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _uniform2nonuniform_cpu_2d(noncart_data, cart_data, interp_value, interp_index):
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
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_y in range(ywidth):
            idy = yindex[frame, point, i_y]
            valy = yvalue[frame, point, i_y]

            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = valy * xvalue[frame, point, i_x]

                noncart_data[frame, batch, point] += (
                    val * cart_data[frame, batch, idy, idx]
                )

    return noncart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _uniform2nonuniform_cpu_3d(noncart_data, cart_data, interp_value, interp_index):
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
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
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

                    noncart_data[frame, batch, point] += (
                        val * cart_data[frame, batch, idz, idy, idx]
                    )

    return noncart_data


# %% GPU
if torch.cuda.is_available():

    @nb.cuda.jit()
    def _uniform2nonuniform_gpu_1d(noncart_data, cart_data, interp_value, interp_index):
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

            # gather data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                noncart_data[frame, batch, point] += val * cart_data[frame, batch, idx]

        return noncart_data

    @nb.cuda.jit()
    def _uniform2nonuniform_gpu_2d(noncart_data, cart_data, interp_value, interp_index):
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

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    noncart_data[frame, batch, point] += (
                        val * cart_data[frame, batch, idy, idx]
                    )

        return noncart_data

    @nb.cuda.jit()
    def _uniform2nonuniform_gpu_3d(noncart_data, cart_data, interp_value, interp_index):
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

                        noncart_data[frame, batch, point] += (
                            val * cart_data[frame, batch, idz, idy, idx]
                        )

        return noncart_data
