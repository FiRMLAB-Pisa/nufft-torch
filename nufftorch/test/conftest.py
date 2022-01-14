""" Configuration utils for test suite. """
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import itertools
import pytest

import torch

# %% test parameters


def _img_size_list():
    return [8]


def _dim_list():
    return [2, 3]


def _device_list():
    devices = ['cpu']

    # if torch.cuda.is_available():
    #     devices.append('cuda')

    return devices


def _data_dtype_list():
    return [torch.complex64]


def _coord_type_list():
    return [torch.float32]


def _basis_dtype_list():
    return [torch.float32, torch.complex64]


def _testing_multiple_coils():
    return [1, 2]


def _testing_multiple_echoes():
    return [1, 2]


def _testing_multiple_slices():
    return [1, 2]


def _testing_multiple_frames():
    return [1, 2]


# %% Utils


class _Utils:

    @staticmethod
    def normalize(data_in, ndim):
        """Normalize Input between 0 and 1."""
        center = int(data_in.shape[-1] // 2)
        
        if ndim == 2:
            scale = torch.abs(data_in[..., center, center])
            scale = scale[..., None, None]
            
        if ndim == 3:
            scale = torch.abs(data_in[..., center, center, center])
            scale = scale[..., None, None, None]

        return data_in / scale


@pytest.fixture
def utils():
    """Wrapper to use Utils as Pytest fixture."""
    return _Utils

# %% data generators


class _ktSpaceTrajectory():

    def __init__(self, ndim, dtype, nframes, npix):

        # build coordinates
        nodes = torch.arange(npix) - (npix // 2)

        if ndim == 2:
            x_i, y_i = torch.meshgrid(nodes, nodes, indexing='ij')
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            coord = torch.stack((x_i, y_i), axis=-1).to(dtype)

        elif ndim == 3:
            x_i, y_i, z_i = torch.meshgrid(nodes, nodes, nodes, indexing='ij')
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            z_i = z_i.flatten()
            coord = torch.stack((x_i, y_i, z_i), axis=-1).to(dtype)

        coord = torch.repeat_interleave(coord[None, :, :], nframes, axis=0)

        # reshape coordinates and build dcf / matrix size
        self.coordinates = coord
        
        if ndim == 2:
            self.density_comp_factor = torch.ones(self.coordinates.shape[:-1], dtype=dtype)[:, None, None, None, :]
            
        elif ndim == 3:
            self.density_comp_factor = torch.ones(self.coordinates.shape[:-1], dtype=dtype)[:, None, None, :]

        self.acquisition_matrix = npix


def _image(ndim, device_id, dtype, nframes, nechoes, ncoils, nslices, npix):

    center = int(npix // 2)

    if ndim == 2:
        img = torch.zeros((nframes, nechoes, ncoils, nslices, npix, npix),
                          dtype=dtype, device=device_id)
        img[:, :, :, :, center, center] = 1

    elif ndim == 3:
        img = torch.zeros((nframes, nechoes, ncoils, npix, npix, npix),
                          dtype=dtype, device=device_id)
        img[:, :, :, center, center, center] = 1

    return img


def _kt_space_data(ndim, device_id, dtype, nframes, nechoes, ncoils, nslices, npix):

    if ndim == 2:
        data = torch.ones((nframes, nechoes, ncoils, nslices, (npix**ndim)),
                          dtype=dtype, device=device_id)

    elif ndim == 3:
        data = torch.ones((nframes, nechoes, ncoils, (npix**ndim)),
                          dtype=dtype, device=device_id)

    return data


def _lowrank_subspace_projection(dtype, nframes):
    return torch.eye(nframes, dtype=dtype)


# %% parametrized cases


def _get_noncartesian_params(lowrank: bool = False,  # pylint: disable=too-many-locals
                             selfadjoint: bool = False):

    # get input argument combinations
    args = [_dim_list(),
            _device_list(),
            _data_dtype_list(),
            _coord_type_list(),
            _testing_multiple_frames(),
            _testing_multiple_echoes(),
            _testing_multiple_coils(),
            _testing_multiple_slices(),
            _img_size_list()]

    # add lowrank basis if enabled
    if lowrank is True:
        args.append(_basis_dtype_list())

    # get combinations
    args = list(itertools.product(*args))

    # build test data
    test_data = []
    
    for arg in args:
        ndim = arg[0]
        device_id = arg[1]
        data_type = arg[2]
        coord_type = arg[3]
        nframes = arg[4]
        nechoes = arg[5]
        ncoils = arg[6]
        nslices = arg[7]
        npix = arg[8]

        tmp = [ndim, device_id,
               _image(ndim, device_id, data_type, nframes,
                      nechoes, ncoils, nslices, npix),
               _ktSpaceTrajectory(ndim, coord_type, nframes, npix)
               ]
        
        if selfadjoint is False:
            tmp.append(_kt_space_data(
                ndim, device_id, data_type, nframes, nechoes, ncoils, nslices, npix))
            
        if lowrank is True:
            basis_type = arg[9]
            tmp.append(_lowrank_subspace_projection(basis_type, nframes))
            
        test_data.append(tmp)

    return test_data
