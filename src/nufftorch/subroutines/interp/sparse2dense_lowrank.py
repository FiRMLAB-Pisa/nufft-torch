import numba as nb
import torch


@nb.njit(fastmath=True, parallel=True, cache=True)  # pragma: no cover
def _sparse2dense_lowrank_cpu_1d(cart_data, noncart_data, interp_index, basis):
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                # spread data within kernel radius
                idx = interp_index[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                cart_data[coeff, batch, idx] += (
                    basis[coeff, frame] * noncart_data[frame, batch, point]
                )

    return cart_data


if torch.cuda.is_available():

    @nb.cuda.jit()
    def _sparse2dense_lowrank_gpu_1d(cart_data, noncart_data, interp_index, basis):
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

            # spread data within kernel radius
            idx = interp_index[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while spreading data
            for coeff in range(ncoeff):
                nb.cuda.atomic.add(
                    cart_data,
                    (coeff, batch, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

        return cart_data

    @nb.cuda.jit()
    def _sparse2dense_lowrank_gpu_cplx_1d(cart_data, noncart_data, interp_index, basis):
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

            # spread data within kernel radius
            idx = interp_index[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
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

        return cart_data
