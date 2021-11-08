import itertools
import pytest
import numpy as np

import sigpy

from qti._config import *
from qti.fourier_transform.interp import ViewSharing

#%% test parameters
def img_size_list():
    return [3, 4]

def device_list():
    devices = [-1]
    
    if sigpy.config.cupy_enabled:
        devices.append(0)

    return devices

def dim_list():
    return [1, 2, 3]

def dtype_list():
    return [np.float32, np.float64, np.complex64, np.complex128]

def coord_type_list():
    return [np.float32, np.float64]

def share_mode_list():
    return ["sliding-window", "view-sharing"]

def share_type_list():
    return [np.int16, np.int32, np.int64]

def sampling_type_list():
    return [np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]

def testing_multiple_coils():
    return [1, 2, 3]

def testing_multiple_slices():
    return [1, 2, 3]

def testing_multiple_frames():
    return [1, 5, 10]

#%% fixtures
@pytest.fixture
def testing_tol():
    return 1e-01


#%% Utils
class Utils:
    
    @staticmethod
    def normalize(input):
        scale = np.max(np.abs(input.ravel()))
        return input / scale
    
@pytest.fixture
def utils():
    return Utils

#%% k-space related objects
class kt_sampling_mask():
    
    def __init__(self, ndim, type, nframes, npix):
        
        # build coordinates
        locations = np.arange(npix)
        
        if ndim == 1:
            xi = locations
            mask = xi[...,np.newaxis].astype(type)

        elif ndim == 2:
            xi,yi = np.meshgrid(locations, locations)
            xi = xi.flatten()
            yi = yi.flatten()
            mask = np.stack((xi, yi),axis=-1).astype(type)
            
        else:
            xi,yi, zi = np.meshgrid(locations, locations, locations)
            xi = xi.flatten()
            yi = yi.flatten()
            zi = zi.flatten()
            mask = np.stack((xi, yi, zi),axis=-1).astype(type)
        
        mask = np.repeat(mask[np.newaxis,:,:], nframes, axis=0)
        
        # reshape coordinates and build dcf / matrix size
        self.mask = mask
        self.acquisition_matrix = npix

        
class k_sampling_mask(kt_sampling_mask):
    
    def __init__(self, ndim, type, npix):
        
        super().__init__(ndim, type, 1, npix)
        
        self.mask = self.mask[0]
     
        
class kt_space_trajectory():
    
    def __init__(self, ndim, type, nframes, npix):
        
        # build coordinates
        nodes = np.arange(npix) - (npix // 2)

        if ndim == 1:
            xi = nodes
            coord = xi[...,np.newaxis].astype(type)
            
        elif ndim == 2:
            xi, yi = np.meshgrid(nodes, nodes)
            xi = xi.flatten()
            yi = yi.flatten()
            coord = np.stack((xi, yi),axis=-1).astype(type)
            
        else:
            xi, yi, zi = np.meshgrid(nodes, nodes, nodes)
            xi = xi.flatten()
            yi = yi.flatten()
            zi = zi.flatten()
            coord = np.stack((xi, yi, zi),axis=-1).astype(type)
        
        coord = np.repeat(coord[np.newaxis,:,:], nframes, axis=0)
        
        # reshape coordinates and build dcf / matrix size
        self.coordinates = coord 
        self.density_comp_factor = np.ones(self.coordinates.shape[:-1], dtype=type)
        self.acquisition_matrix = npix
   
        
class k_space_trajectory(kt_space_trajectory):
    
    def __init__(self, ndim, type, npix):
        
        super().__init__(ndim, type, 1, npix)
        
        self.coordinates = self.coordinates[0]
        self.density_comp_factor = self.density_comp_factor[0]

           
def kt_space_data(ndim, device_id, type, nframes, ncoils, nslices, npix):
    
    if ndim == 3:
        data = sigpy.to_device(np.ones((nframes, ncoils, (npix**ndim)), dtype=type), device_id)
        
    else:
        data = sigpy.to_device(np.ones((nframes, ncoils, nslices, (npix**ndim)), dtype=type), device_id)
        
    return data


def k_space_data(ndim, device_id, type, ncoils, nslices, npix):
    
    if ndim == 3:
        data = sigpy.to_device(np.ones((ncoils, (npix**ndim)), dtype=type), device_id)
        
    else:
        data = sigpy.to_device(np.ones((ncoils, nslices, (npix**ndim)), dtype=type), device_id)
        
    return data

def lowrank_subspace_projection(type, nframes):
    return np.eye(nframes, dtype=type)

class share_object(ViewSharing):
    
    def __init__(self, share_type, ndim, type, nframes, npix):
          
        self.share_type = share_type
        self.index = np.zeros((nframes, (npix**ndim), 2), dtype=type)
        
#%% image-space related objects
def image_1d(device_id, type, ncoils, nslices, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((ncoils, nslices, npix), dtype=type)
    img[:,:,center] = 1
    img = sigpy.to_device(img, device_id)
            
    return img


def image_2d(device_id, type, ncoils, nslices, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((ncoils, nslices, npix, npix), dtype=type)
    img[:,:,center,center] = 1
    img = sigpy.to_device(img, device_id)
           
    return img


def image_3d(device_id, type, ncoils, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((ncoils, npix, npix, npix), dtype=type)
    img[:,center,center,center] = 1
    img = sigpy.to_device(img, device_id)
            
    return img

          
def image_1d_t(device_id, type, nframes, ncoils, nslices, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((nframes, ncoils, nslices, npix), dtype=type)
    img[:,:,:,center] = 1
    img = sigpy.to_device(img, device_id)
            
    return img


def image_2d_t(device_id, type, nframes, ncoils, nslices, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((nframes, ncoils, nslices, npix, npix), dtype=type)
    img[:,:,:,center,center] = 1
    img = sigpy.to_device(img, device_id)
           
    return img


def image_3d_t(device_id, type, nframes, ncoils, npix):
    
    # calculate image center
    center = npix // 2
    
    # build image
    img = np.zeros((nframes, ncoils, npix, npix, npix), dtype=type)
    img[:,:,center,center,center] = 1
    img = sigpy.to_device(img, device_id)
            
    return img

#%% parametrized cases
def get_params_1d_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_1d(device_id, data_type, ncoils, nslices, npix),
                          k_space_data(1, device_id, data_type, ncoils, nslices, npix),
                          k_space_trajectory(1, coord_type, npix)))
    
    return test_data


def get_params_2d_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix),
                          k_space_data(2, device_id, data_type, ncoils, nslices, npix),
                          k_space_trajectory(2, coord_type, npix)))
    
    return test_data


def get_params_3d_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_3d(device_id, data_type, ncoils, npix),
                          k_space_data(3, device_id, data_type, ncoils, 1, npix),
                          k_space_trajectory(3, coord_type, npix)))
    
    return test_data


def get_params_1d_t_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_nufft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_nufft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_nufft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_nufft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix)))
    
    return test_data

def get_params_1d_t_nufft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_nufft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_nufft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_t_nufft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_nufft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_nufft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_1d(device_id, data_type, ncoils, nslices, npix), 
                          k_space_data(1, device_id, data_type, ncoils, nslices, npix),
                          k_sampling_mask(1, sampling_type, npix)))
    
    return test_data


def get_params_2d_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix),
                          k_space_data(2, device_id, data_type, ncoils, nslices, npix),
                          k_sampling_mask(2, sampling_type, npix)))
    
    return test_data


def get_params_3d_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_3d(device_id, data_type, ncoils, npix),
                          k_space_data(3, device_id, data_type, ncoils, 1, npix),
                          k_sampling_mask(3, sampling_type, npix)))
    
    return test_data


def get_params_1d_t_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes,ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_fft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_fft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_fft_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_fft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_fft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_fft_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes,ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_t_fft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(1, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_fft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_data(2, device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_fft_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_data(3, device_id, data_type, nframes, ncoils, 1, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_1d(device_id, data_type, ncoils, nslices, npix),
                          k_space_trajectory(1, coord_type, npix)))
    
    return test_data


def get_params_2d_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix),
                          k_space_trajectory(2, coord_type, npix)))
    
    return test_data


def get_params_3d_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        ncoils = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_3d(device_id, data_type, ncoils, npix),
                          k_space_trajectory(3, coord_type, npix)))
    
    return test_data


def get_params_1d_t_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_nufft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_nufft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_nufft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_nufft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix)))
    
    return test_data

def get_params_1d_t_nufft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_nufft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_nufft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_t_nufft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(1, coord_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_nufft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_space_trajectory(2, coord_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_nufft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    coord_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        coord_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_space_trajectory(3, coord_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix), 
                          k_sampling_mask(1, sampling_type, npix)))
    
    return test_data


def get_params_2d_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        nslices = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix),
                          k_sampling_mask(2, sampling_type, npix)))
    
    return test_data


def get_params_3d_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        ncoils = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_3d(device_id, data_type, ncoils, npix),
                          k_sampling_mask(3, sampling_type, npix)))
    
    return test_data


def get_params_1d_t_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_fft_selfadjoint():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes,ncoils, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_fft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix)))
    
    return test_data


def get_params_2d_t_fft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix)))
    
    return test_data


def get_params_3d_t_fft_selfadjoint_viewshare():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix)))
    
    return test_data


def get_params_1d_t_fft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]

        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_fft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        nslices = args[i][5]
        npix = args[i][6]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_fft_selfadjoint_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        nframes = args[i][3]
        ncoils = args[i][4]
        npix = args[i][5]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes,ncoils, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_t_fft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]

        
        test_data.append((device_id,
                          image_1d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(1, sampling_type, nframes, npix),
                          share_object(share_mode, 1, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_2d_t_fft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        nslices = args[i][7]
        npix = args[i][8]
        
        test_data.append((device_id,
                          image_2d_t(device_id, data_type, nframes, ncoils, nslices, npix),
                          kt_sampling_mask(2, sampling_type, nframes, npix),
                          share_object(share_mode, 2, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_3d_t_fft_selfadjoint_viewshare_lowrank():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    sampling_type_list(),
                                    share_mode_list(),
                                    share_type_list(),
                                    testing_multiple_frames(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        sampling_type = args[i][2]
        share_mode = args[i][3]
        share_type = args[i][4]
        nframes = args[i][5]
        ncoils = args[i][6]
        npix = args[i][7]
        
        test_data.append((device_id,
                          image_3d_t(device_id, data_type, nframes, ncoils, npix),
                          kt_sampling_mask(3, sampling_type, nframes, npix),
                          share_object(share_mode, 3, share_type, nframes, npix),
                          lowrank_subspace_projection(data_type, nframes)))
    
    return test_data


def get_params_1d_simple_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        ncoils = args[i][2]
        nslices = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_1d(device_id, data_type, ncoils, nslices, npix), 
                          k_space_data(1, device_id, data_type, ncoils, nslices, npix)))
    
    return test_data


def get_params_2d_simple_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    testing_multiple_coils(),
                                    testing_multiple_slices(),
                                    img_size_list()]))
        
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        ncoils = args[i][2]
        nslices = args[i][3]
        npix = args[i][4]
        
        test_data.append((device_id,
                          image_2d(device_id, data_type, ncoils, nslices, npix),
                          k_space_data(2, device_id, data_type, ncoils, nslices, npix)))
    
    return test_data


def get_params_3d_simple_fft():
    
    args = list(itertools.product(*[device_list(),
                                    dtype_list(),
                                    testing_multiple_coils(),
                                    img_size_list()]))
    
    test_data = []
    
    for i in range(len(args)):
        device_id = args[i][0]
        data_type = args[i][1]
        ncoils = args[i][2]
        npix = args[i][3]
        
        test_data.append((device_id,
                          image_3d(device_id, data_type, ncoils, npix),
                          k_space_data(3, device_id, data_type, ncoils, 1, npix)))
    
    return test_data

