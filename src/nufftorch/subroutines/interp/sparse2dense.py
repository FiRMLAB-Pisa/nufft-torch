"""Routines to perform zero-filling of (Cartesian) under-sampled data."""
import numba as nb
import torch


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _sparse2dense_cpu(cart_data, noncart_data, interp_index):
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            idx = interp_index[frame, point]
            cart_data[frame, batch, idx] += noncart_data[frame, batch, point]

    return cart_data


if torch.cuda.is_available():

    @nb.cuda.jit()
    def _sparse2dense_gpu(cart_data, noncart_data, interp_index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            idx = interp_index[frame, point]
            nb.cuda.atomic.add(
                cart_data,
                (frame, batch, idx),
                noncart_data[frame, batch, point],
            )

        return cart_data

    @nb.cuda.jit()
    def _sparse2dense_gpu_cplx(cart_data, noncart_data, interp_index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within
            idx = interp_index[frame, point]
            nb.cuda.atomic.add(
                cart_data.real,
                (frame, batch, idx),
                noncart_data[frame, batch, point].real,
            )
            nb.cuda.atomic.add(
                cart_data.imag,
                (frame, batch, idx),
                noncart_data[frame, batch, point].imag,
            )

        return cart_data
