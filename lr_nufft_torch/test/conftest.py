""" Configuration utils for test suite. """
import itertools
import pytest

import torch

# %% test parameters


def img_size_list():
    return [3, 4]


def device_list():
    devices = ['cpu']

    if torch.cuda.is_available() is True:
        devices.append('cuda')

    return devices


def dim_list():
    return [2, 3]


def dtype_list():
    return [torch.float32, torch.float64, torch.complex64, torch.complex128]


def coord_type_list():
    return [torch.float32, torch.float64]


def testing_multiple_coils():
    return [1, 2, 3]


def testing_multiple_echoes():
    return [1, 2, 3]


def testing_multiple_slices():
    return [1, 2, 3]


def testing_multiple_frames():
    return [1, 2, 3]

# %% fixtures


@pytest.fixture
def testing_tol():
    return 1e-01

# %% Utils


class Utils:

    @staticmethod
    def normalize(input):
        scale = torch.max(torch.abs(input.ravel()))
        return input / scale


@pytest.fixture
def utils():
    return Utils

# %% k-space related objects


class kt_space_trajectory():
    """ Generate kt-space trajectory."""

    def __init__(self, ndim, dtype, nframes, npix):

        # build coordinates
        nodes = torch.arange(npix) - (npix // 2)

        if ndim == 1:
            xi = nodes
            coord = xi[..., None].astype(type)

        elif ndim == 2:
            xi, yi = torch.meshgrid(nodes, nodes)
            xi = xi.flatten()
            yi = yi.flatten()
            coord = torch.stack((xi, yi), axis=-1).astype(type)

        else:
            xi, yi, zi = torch.meshgrid(nodes, nodes, nodes)
            xi = xi.flatten()
            yi = yi.flatten()
            zi = zi.flatten()
            coord = torch.stack((xi, yi, zi), axis=-1).astype(type)

        coord = torch.repeat(coord[None, :, :], nframes, axis=0)

        # reshape coordinates and build dcf / matrix size
        self.coordinates = coord
        self.density_comp_factor = torch.ones(
            self.coordinates.shape[:-1], dtype=dtype)
        self.acquisition_matrix = npix


def kt_space_data(ndim, device_id, dtype, nframes, ncoils, nslices, npix):
    """ Generate kt-space data."""
    if ndim == 3:
        data = torch.ones((nframes, ncoils, (npix**ndim)),
                          dtype=dtype, device=device_id)

    else:
        data = torch.ones((nframes, ncoils, nslices, (npix**ndim)),
                          dtype=dtype, device=device_id)

    return data


def lowrank_subspace_projection(dtype, nframes):
    """ Generate low-rank subspace basis data."""
    return torch.eye(nframes, dtype=dtype)


# %% image-space related objects
def image_2d(device_id, dtype, nframes, ncoils, nslices, npix):
    """ Generate image-space data."""
    # calculate image center
    center = npix // 2

    # build image
    img = torch.zeros((nframes, ncoils, nslices, npix, npix),
                      dtype=dtype, device=device_id)
    img[:, :, :, center, center] = 1

    return img


def image_3d(device_id, dtype, nframes, ncoils, npix):
    """ Generate image-space data."""
    # calculate image center
    center = npix // 2

    # build image
    img = torch.zeros((nframes, ncoils, npix, npix, npix),
                      dtype=dtype, device=device_id)
    img[:, :, center, center, center] = 1

    return img

# %% parametrized cases


def get_params_2d_nufft():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        nslices = arg[5]
        npix = arg[6]

        test_data.append((device_id,
                          image_2d(device_id, data_type, nframes,
                                   ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type,
                                        nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix)))

    return test_data


def get_params_3d_nufft():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        npix = arg[5]

    test_data.append((device_id,
                      image_3d(device_id, data_type,
                               nframes, ncoils, npix),
                      kt_space_data(3, device_id, data_type,
                                    nframes, ncoils, 1, npix),
                      kt_space_trajectory(3, coord_type, nframes, npix)))

    return test_data


def get_params_2d_nufft_lowrank():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        nslices = arg[5]
        npix = arg[6]

    test_data.append((device_id,
                      image_2d(device_id, data_type, nframes,
                               ncoils, nslices, npix),
                      kt_space_data(2, device_id, data_type,
                                    nframes, ncoils, nslices, npix),
                      kt_space_trajectory(2, coord_type, nframes, npix),
                      lowrank_subspace_projection(data_type, nframes)))

    return test_data


def get_params_3d_nufft_lowrank():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        npix = arg[5]

        test_data.append((device_id,
                          image_3d(device_id, data_type,
                                   nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type,
                                        nframes, ncoils, 1, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))

    return test_data


def get_params_2d_nufft_selfadjoint():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        nslices = arg[5]
        npix = arg[6]

        test_data.append((device_id,
                          image_2d(device_id, data_type, nframes,
                                   ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix)))

    return test_data


def get_params_3d_nufft_selfadjoint():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        npix = arg[5]

        test_data.append((device_id,
                          image_3d(device_id, data_type,
                                   nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix)))

    return test_data


def get_params_2d_nufft_selfadjoint_lowrank():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        nslices = arg[5]
        npix = arg[6]

        test_data.append((device_id,
                          image_2d(device_id, data_type, nframes,
                                   ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))

    return test_data


def get_params_3d_nufft_selfadjoint_lowrank():

    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))

    test_data = []

    for arg in args:
        device_id = arg[0]
        data_type = arg[1]
        coord_type = arg[2]
        nframes = arg[3]
        ncoils = arg[4]
        npix = arg[5]

        test_data.append((device_id,
                          image_3d(device_id, data_type,
                                   nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))

    return test_data

