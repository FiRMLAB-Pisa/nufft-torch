"""Routines to perform sampling of (Cartesian) under-sampled data with simultaneous low-rank subspace backprojection."""
import numba as nb
import torch


# %% CPU
@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _dense2sparse_lowrank_cpu(noncart_data, cart_data, interp_index, adjoint_basis):
    # get sizes
    ncoeff, batch_size, _ = cart_data.shape
    nframes = noncart_data.shape[0]
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

        # do adjoint low rank projection (low-rank subspace -> time domain)
        # while gathering data
        for coeff in range(ncoeff):
            noncart_data[frame, batch, point] += (
                adjoint_basis[frame, coeff] * cart_data[coeff, batch, idx]
            )

    return noncart_data


# %% GPU
if torch.cuda.is_available():

    @nb.cuda.jit()
    def _dense2sparse_lowrank_gpu(noncart_data, cart_data, interp_index, adjoint_basis):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
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

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while gathering data
            for coeff in range(ncoeff):
                noncart_data[frame, batch, point] += (
                    adjoint_basis[frame, coeff] * cart_data[coeff, batch, idx]
                )

        return noncart_data
