# -*- coding: utf-8 -*-
"""
Factory routines to build NUFFT/Toeplitz objects.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
# pylint: disable=line-too-long

import numpy as np
import torch

from nufftorch.src import _cpu
from nufftorch.src._subroutines import (Apodize,
                                        BackendBridge,
                                        Crop,
                                        FFT,
                                        Grid,
                                        IFFT,
                                        Utils,
                                        ZeroPad)


class AbstractFactory:
    """Base class for NUFFT and Toeplitz factory classes."""

    def _parse_coordinate_size(self, coord):

        # inspect input coordinates
        self.nframes = coord.shape[0]
        pts_shape = coord.shape[1:-1]
        self.ndim = coord.shape[-1]

        # calculate total number of point per frame and total number of points
        self.npts = np.prod(pts_shape)
        self.coord_length = self.nframes*self.npts

    def _reformat_coordinates(self, coord):
        return coord.reshape([self.coord_length, self.ndim]).T


class NUFFTFactory(AbstractFactory):
    """Functions to prepare NUFFT operator."""

    def __call__(self, coord, shape, width, oversamp, basis, sharing_width, device, threadsperblock):
        """Actual pre-computation routine."""

        # get parameters
        ndim = coord.shape[-1]

        # get arrays
        width, shape = Utils._scalars2arrays(ndim, width, shape)

        # get kernel shape parameter
        beta = Utils._beatty_parameter(width, oversamp)

        # get grid shape
        grid_shape = Utils._get_oversampled_shape(oversamp, shape)

        # adjust coordinates
        coord = Utils._scale_coord(coord, shape, oversamp)

        # compute interpolator
        kernel_dict = self._prepare_kernel(
            coord, grid_shape, width, beta, basis, sharing_width, device)

        # device_dict
        device_dict = {'device': device, 'threadsperblock': threadsperblock}

        # create interpolator dict
        interpolator = {'ndim': ndim,
                        'oversamp': oversamp,
                        'width': width,
                        'beta': beta,
                        'kernel_dict': kernel_dict,
                        'device_dict': device_dict}

        return interpolator

    def _prepare_kernel(self, coord, shape, width, beta, basis, sharing_width, device):

        # parse size
        self._parse_coordinate_size(coord)

        # reformat coordinates
        coord = self._reformat_coordinates(coord)

        # calculate interpolator
        kernel_tuple = self._prepare_sparse_coefficient_matrix(
            coord, shape, width, beta, device)

        # adjust basis
        if basis is not None:
            # build basis adjoint
            basis_adjoint = basis.conj().T

            # convert to numpy or numba cuda array
            basis = Utils.pytorch2numba(basis.to(device))
            basis_adjoint = Utils.pytorch2numba(basis_adjoint.to(device))
        else:
            basis_adjoint = None

        # prepare kernel dictionary
        kernel_dict = {'sparse_coefficients': kernel_tuple,
                       'coord_shape': coord.shape,
                       'grid_shape': shape,
                       'width': width,
                       'basis': basis,
                       'basis_adjoint': basis_adjoint,
                       'sharing_width': sharing_width}

        return kernel_dict

    def _preallocate_sparse_coefficient_matrix(self, width):
        index = []
        value = []

        for i in range(self.ndim):
            # kernel value
            empty_value = torch.zeros(
                (self.coord_length, width[i]), dtype=torch.float32)
            value.append(empty_value)

            # kernel index
            empty_index = torch.zeros(
                (self.coord_length, width[i]), dtype=torch.int32)
            index.append(empty_index)

        return value, index

    def _reformat_sparse_coefficients(self, value, index, width, device):
        for i in range(self.ndim):
            value[i] = value[i].reshape(
                [self.nframes, self.npts, width[i]]).to(device)
            index[i] = index[i].reshape(
                [self.nframes, self.npts, width[i]]).to(device)
            value[i], index[i] = BackendBridge.pytorch2numba(
                value[i], index[i])

    def _prepare_sparse_coefficient_matrix(self, coord, shape, width, beta, device):

        # preallocate interpolator
        value, index = self._preallocate_sparse_coefficient_matrix(width)

        # actual computation
        for i in range(self.ndim):
            val, ind, coo = BackendBridge.pytorch2numba(
                value[i], index[i], coord[i])
            _cpu._prepare_sparse_coefficient_matrix(
                val, ind, coo, beta[i], shape[i])


        # reformat coefficients
        self._reformat_sparse_coefficients(value, index, width, device)

        # pack kernel tuple
        kernel_tuple = (tuple(value), tuple(index), shape)

        return kernel_tuple


class AbstractToeplitzFactory(AbstractFactory):
    """Base class for Cartesian and Non-Cartesian Factory classes."""

    def _process_basis(self, basis, device):

        # inspect basis to determine data type
        if basis is not None:
            self.islowrank = True
            self.isreal = not torch.is_complex(basis)
            self.ncoeff = basis.shape[0]
        else:
            self.islowrank = False
            self.isreal = False
            self.ncoeff = self.nframes

        if self.isreal:
            self.dtype = torch.float32
        else:
            self.dtype = torch.complex64

        # if spatio-temporal basis is provided, check reality and offload to device
        if self.islowrank:
            basis = basis.to(self.dtype).to(device)
            basis_adjoint = basis.conj().T
        else:
            basis_adjoint = None

        return basis, basis_adjoint

    def _preallocate_unitary_kspace_data(self, basis_adjoint, device):

        if self.islowrank:
            # initialize temporary arrays
            ones = torch.ones((self.nframes, self.ncoeff, self.npts),
                              dtype=torch.complex64, device=device)
            ones = ones * basis_adjoint[:, :, None]

        else:
            # initialize temporary arrays
            ones = torch.ones((self.nframes, self.npts),
                              dtype=torch.complex64, device=device)

        return ones

    def _post_process_mtf(self, mtf):

        # keep only real part if basis is real
        if self.isreal:
            mtf = mtf.real

        # fftshift kernel to accelerate computation
        mtf = torch.fft.ifftshift(mtf, dim=list(range(-self.ndim, 0)))

        if self.islowrank:
            mtf = mtf.reshape(*mtf.shape[:2], np.prod(mtf.shape[2:]))
        else:
            mtf = mtf[:, None, ...]

        # normalize
        # mtf /= torch.quantile(mtf, 0.95)

        # remove NaN
        mtf = torch.nan_to_num(mtf)

        # transform to numba
        mtf = BackendBridge.pytorch2numba(mtf)

        return mtf


class NonCartesianToeplitzFactory(AbstractToeplitzFactory):
    """Functions to prepare Toeplitz kernel for Non-Cartesian sampling."""

    def __call__(self, coord, shape, prep_osf, comp_osf, width,
                 basis, sharing_width, device, threadsperblock, dcf):
        """Actual pre-computation routine."""
        # clone coordinates and dcf to avoid modifications
        coord = coord.clone()

        # get parameters
        ndim = coord.shape[-1]

        # get arrays
        width, shape = Utils._scalars2arrays(ndim, width, shape)

        # get kernel shape parameter
        beta = Utils._beatty_parameter(width, prep_osf)

        # get grid shape for preparation
        grid_shape = Utils._get_oversampled_shape(shape, prep_osf)

        # adjust coordinates
        coord = Utils._scale_coord(coord, shape, prep_osf)

        # device_dict
        device_dict = {'device': device, 'threadsperblock': threadsperblock}

        # actual kernel precomputation
        mtf = self._prepare_kernel(
            coord, grid_shape, prep_osf, comp_osf, width, beta, basis, sharing_width, device_dict, dcf)

        # get grid shape for computation
        grid_shape = Utils._get_oversampled_shape(shape, comp_osf)

        return {'mtf': mtf, 'islowrank': self.islowrank, 'device_dict': device_dict,
                'ndim': ndim, 'grid_shape': grid_shape}

    def _prepare_kernel(self, coord, shape, prep_osf, comp_osf, width, beta, basis, sharing_width, device_dict, dcf):
        """Core kernel computation sub-routine."""
        # parse size
        self._parse_coordinate_size(coord)

        # reformat coordinates
        coord = self._reformat_coordinates(coord)

        # calculate interpolator
        mtf = self._prepare_coefficient_matrix(
            coord, shape, prep_osf, width, basis, sharing_width, device_dict, dcf)

        # resample mtf
        mtf = self._resample_mtf(
            mtf, prep_osf, comp_osf, width, beta, device_dict['device'])

        # post process mtf
        mtf = self._post_process(mtf)

        return mtf

    def _prepare_coefficient_matrix(self, coord, shape, oversamp, width, basis, sharing_width, device_dict, dcf):
        # process basis
        basis, basis_adjoint = self._process_basis(
            basis, device_dict['device'])

        # prepare dcf
        dcf = self._prepare_dcf(dcf, device_dict['device'])

        # prepare unitary data for MTF/PSF estimation
        ones = self._preallocate_unitary_kspace_data(
            basis_adjoint, device_dict['device'])

        # calculate modulation transfer function
        mtf = self._gridding(ones, coord, shape, width,
                             oversamp, basis, sharing_width, device_dict)

        return mtf

    def _prepare_dcf(self, dcf, device):

        # if dcf are not provided, assume uniform sampling density
        if dcf is None:
            dcf = torch.ones((self.nframes, self.npts),
                             torch.float32, device=device)
        else:
            dcf = dcf.clone()
            dcf = dcf.to(device)

        if dcf.ndim > 1 and not self.islowrank:
            dcf = dcf[:, None, :]

        return dcf

    def _gridding(self, ones, coord, shape, width, oversamp, basis, sharing_width, device):
        # prepare NUFFT object
        nufft_dict = NUFFTFactory()(coord, shape, width, oversamp,
                                    basis, sharing_width, device)

        # compute mtf
        return Grid(device)(ones, nufft_dict['kernel_dict'])

    def _resample_mtf(self, mtf, prep_osf, comp_osf, width, beta, device):

        # IFFT
        psf = IFFT()(mtf, axes=range(-self.ndim, 0), norm='ortho')

        # Crop
        psf = Crop(prep_osf, psf.shape[:-self.ndim])(psf)

        # Apodize
        Apodize(self.ndim, prep_osf, width, beta, device)(psf)

        # Zero-pad
        psf = ZeroPad(comp_osf, psf.shape[-self.ndim:])(psf)

        # FFT
        mtf = FFT()(psf, axes=range(-self.ndim, 0), norm='ortho')

        return mtf
