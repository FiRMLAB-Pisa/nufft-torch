"""Routines to perform gridding of (non-Cartesian) under-sampled data with simultaneous view-sharing and low-rank subspace projection."""
import numba as nb
import torch


#%% CPU
@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_lowrank_vs_cpu_1d(
    cart_data,
    noncart_data,
    interp_value,
    interp_index,
    basis,
    window_width,
):
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = interp_index
    xvalue = interp_value

    # get interpolator width
    xwidth = xindex.shape[-1]

    # get half window width
    hwidth = window_width // 2

    # account for window
    nframes -= hwidth

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            frame += hwidth
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                # spread data within kernel radius
                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = xvalue[frame, point, i_x]

                    # iterate over sharing window
                    for share_idx in range(-hwidth, hwidth):
                        # simultaneous sharing and low rank projection
                        cart_data[coeff, batch, idx] += (
                            val
                            * basis[coeff, frame + share_idx]
                            * noncart_data[frame, batch, point]
                        )

    return cart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_lowrank_vs_cpu_2d(
    cart_data,
    noncart_data,
    interp_value,
    interp_index,
    basis,
    window_width,
):
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

    # get half window width
    hwidth = window_width // 2

    # account for window
    nframes -= hwidth

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            frame += hwidth
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # iterate over sharing window
                        for share_idx in range(-hwidth, hwidth):
                            # simultaneous sharing and low rank projection
                            cart_data[coeff, batch, idy, idx] += (
                                val
                                * basis[coeff, frame + share_idx]
                                * noncart_data[frame, batch, point]
                            )

    return cart_data


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _nonuniform2uniform_lowrank_vs_cpu_3d(
    cart_data,
    noncart_data,
    interp_value,
    interp_index,
    basis,
    window_width,
):
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

    # get half window width
    hwidth = window_width // 2

    # account for window
    nframes -= hwidth

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            frame += hwidth
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

                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # simultaneous sharing and low rank projection
                                cart_data[coeff, batch, idz, idy, idx] += (
                                    val
                                    * basis[coeff, frame + share_idx]
                                    * noncart_data[frame, batch, point]
                                )

    return cart_data

# function table
_nonuniform2uniform_lowrank_vs_cpu = [_nonuniform2uniform_lowrank_vs_cpu_1d, _nonuniform2uniform_lowrank_vs_cpu_2d, _nonuniform2uniform_lowrank_vs_cpu_3d]

#%% GPU
if torch.cuda.is_available():

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_1d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index
        xvalue = interp_value

        # get interpolator width
        xwidth = xindex.shape[-1]

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                # do low rank projection (time domain -> low-rank subspace)
                # while spreading data
                for coeff in range(ncoeff):
                    # iterate over sharing window
                    for share_idx in range(-hwidth, hwidth):
                        # simultaneous sharing and low rank projection
                        nb.cuda.atomic.add(
                            cart_data,
                            (coeff, batch, idx),
                            val
                            * basis[coeff, frame + share_idx]
                            * noncart_data[frame, batch, point],
                        )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_2d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
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

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
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

                    for coeff in range(ncoeff):
                        # iterate over sharing window
                        for share_idx in range(-hwidth, hwidth):
                            # simultaneous sharing and low rank projection
                            nb.cuda.atomic.add(
                                cart_data,
                                (coeff, batch, idy, idx),
                                val
                                * basis[coeff, frame + share_idx]
                                * noncart_data[frame, batch, point],
                            )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_3d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
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

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
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

                        # do low rank projection (time domain -> low-rank subspace)
                        # while gathering data
                        for coeff in range(ncoeff):
                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # simultaneous sharing and low rank projection
                                nb.cuda.atomic.add(
                                    cart_data,
                                    (coeff, batch, idz, idy, idx),
                                    val
                                    * basis[coeff, frame + share_idx]
                                    * noncart_data[frame, batch, point],
                                )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_cplx_1d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index
        xvalue = interp_value

        # get interpolator width
        xwidth = xindex.shape[-1]

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                # do low rank projection (time domain -> low-rank subspace)
                # while spreading data
                for coeff in range(ncoeff):
                    # iterate over sharing window
                    for share_idx in range(-hwidth, hwidth):
                        # simultaneous sharing and low rank projection
                        nb.cuda.atomic.add(
                            cart_data.real,
                            (coeff, batch, idx),
                            (
                                val
                                * basis[coeff, frame + share_idx]
                                * noncart_data[frame, batch, point]
                            ).real,
                        )
                        nb.cuda.atomic.add(
                            cart_data.imag,
                            (coeff, batch, idx),
                            (
                                val
                                * basis[coeff, frame + share_idx]
                                * noncart_data[frame, batch, point]
                            ).imag,
                        )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_cplx_2d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
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

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
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

                    # do low rank projection (time domain -> low-rank subspace)
                    # while spreading data
                    for coeff in range(ncoeff):
                        # iterate over sharing window
                        for share_idx in range(-hwidth, hwidth):
                            # simultaneous sharing and low rank projection
                            nb.cuda.atomic.add(
                                cart_data.real,
                                (coeff, batch, idy, idx),
                                (
                                    val
                                    * basis[coeff, frame + share_idx]
                                    * noncart_data[frame, batch, point]
                                ).real,
                            )
                            nb.cuda.atomic.add(
                                cart_data.imag,
                                (coeff, batch, idy, idx),
                                (
                                    val
                                    * basis[coeff, frame + share_idx]
                                    * noncart_data[frame, batch, point]
                                ).imag,
                            )

        return cart_data

    @nb.cuda.jit()
    def _nonuniform2uniform_lowrank_vs_gpu_cplx_3d(
        cart_data,
        noncart_data,
        interp_value,
        interp_index,
        basis,
        window_width,
    ):
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

        # get half window width
        hwidth = window_width // 2

        # account for window
        nframes -= hwidth

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts) + hwidth
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

                        # do low rank projection (time domain -> low-rank subspace)
                        # while gathering data
                        for coeff in range(ncoeff):
                            # iterate over sharing window
                            for share_idx in range(-hwidth, hwidth):
                                # simultaneous sharing and low rank projection
                                nb.cuda.atomic.add(
                                    cart_data.real,
                                    (coeff, batch, idz, idy, idx),
                                    (
                                        val
                                        * basis[coeff, frame + share_idx]
                                        * noncart_data[frame, batch, point]
                                    ).real,
                                )
                                nb.cuda.atomic.add(
                                    cart_data.imag,
                                    (coeff, batch, idz, idy, idx),
                                    (
                                        val
                                        * basis[coeff, frame + share_idx]
                                        * noncart_data[frame, batch, point]
                                    ).imag,
                                )

        return cart_data
    
    # function table
    _nonuniform2uniform_lowrank_vs_gpu = [_nonuniform2uniform_lowrank_vs_gpu_1d, _nonuniform2uniform_lowrank_vs_gpu_2d, _nonuniform2uniform_lowrank_vs_gpu_3d]
    _nonuniform2uniform_lowrank_vs_gpu_cplx = [_nonuniform2uniform_lowrank_vs_gpu_cplx_1d, _nonuniform2uniform_lowrank_vs_gpu_cplx_2d, _nonuniform2uniform_lowrank_vs_gpu_cplx_3d]

