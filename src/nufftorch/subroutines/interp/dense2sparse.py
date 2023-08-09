"""Routines to perform sampling of (Cartesian) under-sampled data."""
import numba as nb
import torch


# %% CPU
@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _dense2sparse_cpu(noncart_data, cart_data, interp_index):
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data
        idx = interp_index[frame, point]
        noncart_data[frame, batch, point] += cart_data[frame, batch, idx]

    return noncart_data


# %% GPU
if torch.cuda.is_available():

    @nb.cuda.jit()
    def _dense2sparse_gpu(noncart_data, cart_data, interp_index):
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

            # gather data
            idx = interp_index[frame, point]
            noncart_data[frame, batch, point] += cart_data[frame, batch, idx]

        return noncart_data
