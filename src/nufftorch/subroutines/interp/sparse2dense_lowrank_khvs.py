"""Routines to perform zero-filling of (Cartesian) under-sampled data with simultaneous soft-weighted keyhole view-sharing and low-rank subspace projection."""
import numba as nb
import torch


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _sparse2dense_lowrank_khvs_cpu(
    cart_data,
    noncart_data,
    interp_index,
    basis,
    window_width,
    spatial_weight,
    temporal_weight,
):
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

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
                idx = interp_index[frame, point]

                # do low rank projection (time domain -> low-rank subspace)
                # while spreading data
                cart_data[coeff, batch, idx] += (
                    basis[coeff, frame] * noncart_data[frame, batch, point]
                )

                # now share
                # update kernel according to spatial weight
                sw = spatial_weight[idx]

                # iterate over sharing window
                for share_idx in range(-hwidth, hwidth):
                    # select temporal weight for current target frame
                    tw = temporal_weight[frame, hwidth + share_idx]

                    # simultaneous sharing and low rank projection
                    cart_data[coeff, batch, idx] += (
                        sw
                        * tw
                        * basis[coeff, frame + share_idx]
                        * noncart_data[frame, batch, point]
                    )

    return cart_data


if torch.cuda.is_available():

    @nb.cuda.jit()
    def _sparse2dense_lowrank_khvs_gpu(
        cart_data,
        noncart_data,
        interp_index,
        basis,
        window_width,
        spatial_weight,
        temporal_weight,
    ):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

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

            # spread data
            idx = interp_index[frame, point]

            # update kernel according to spatial weight
            sw = spatial_weight[idx]

            # do low rank projection (time domain -> low-rank subspace)
            # while spreading data
            for coeff in range(ncoeff):
                # update frame
                nb.cuda.atomic.add(
                    cart_data,
                    (coeff, batch, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

                # now share
                # iterate over sharing window
                for share_idx in range(-hwidth, hwidth):
                    # select temporal weight for current target frame
                    tw = temporal_weight[frame, hwidth + share_idx]

                    # simultaneous sharing and low rank projection
                    nb.cuda.atomic.add(
                        cart_data,
                        (coeff, batch, idx),
                        sw
                        * tw
                        * basis[coeff, frame + share_idx]
                        * noncart_data[frame, batch, point],
                    )

        return cart_data

    @nb.cuda.jit()
    def _sparse2dense_lowrank_khvs_gpu_cplx(
        cart_data,
        noncart_data,
        interp_index,
        basis,
        window_width,
        spatial_weight,
        temporal_weight,
    ):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

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

            # spread data
            idx = interp_index[frame, point]

            # update kernel according to spatial weight
            sw = spatial_weight[idx]

            # do low rank projection (time domain -> low-rank subspace)
            # while spreading data
            for coeff in range(ncoeff):
                nb.cuda.atomic.add(
                    cart_data.real,
                    (coeff, batch, idx),
                    (basis[coeff, frame] * noncart_data[frame, batch, point]).real,
                )
                nb.cuda.atomic.add(
                    cart_data.imag,
                    (coeff, batch, idx),
                    (basis[coeff, frame] * noncart_data[frame, batch, point]).imag,
                )

                # now share
                # iterate over sharing window
                for share_idx in range(-hwidth, hwidth):
                    # select temporal weight for current target frame
                    tw = temporal_weight[frame, hwidth + share_idx]

                    # simultaneous sharing and low rank projection
                    nb.cuda.atomic.add(
                        cart_data.real,
                        (coeff, batch, idx),
                        (
                            sw
                            * tw
                            * basis[coeff, frame + share_idx]
                            * noncart_data[frame, batch, point]
                        ).real,
                    )
                    nb.cuda.atomic.add(
                        cart_data.imag,
                        (coeff, batch, idx),
                        (
                            sw
                            * tw
                            * basis[coeff, frame + share_idx]
                            * noncart_data[frame, batch, point]
                        ).imag,
                    )

        return cart_data
