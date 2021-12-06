# -*- coding: utf-8 -*-
"""
Utility functions to benchmark lr-nufft-torch.

Compares performance with torchkbnufft.
"""
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

import torchkbnufft as tkbn

import warnings

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)


def create_radial_trajectory(ndims: int, matrix_size: int, nreadouts: int, nframes: int) -> Tuple[Tensor, Tensor]:
    """ Generate golden angle (means) 2D (3D projection) radial trajectory
    i.e. kooshball.

    Args:
        ndims (int): Number of spatial dimensions.
        matrix_size (int): Image number of voxel (assume isotropic FOV).
        nreadouts (int): Number of acquired readouts.
        nframes (int): Number of acquired frames (final trajectory will have
                       nframes and nreadouts // nframes readouts per frame).

    Returns:
        ktraj (tensor): K-space trajectory.
    """
    if ndims == 2:
        ktraj = _create_2d_radial(matrix_size, nreadouts)
    if ndims == 3:
        ktraj = _create_3d_radial(matrix_size, nreadouts)

    # reshape
    ktraj = ktraj[:, :, None, :]
    ktraj = ktraj.reshape([ktraj.shape[0], nreadouts //
                          nframes, nframes, ktraj.shape[-1]])
    ktraj = ktraj.permute(2, 0, 1, -1).to(torch.float32)
    
    # calculate dcf
    if ndims == 2:
        ks = (ktraj**2).sum(axis=-1)**0.5
        dcf = ks
        
    else:
        ks = (ktraj**2).sum(axis=-1)**0.5
        k0 = ks.shape[1] // 2
        dcf0 = 4/3 * np.pi * (ks[:,[k0],:]**3)
        dcf = 4/3 * np.pi * (ks[:,k0+1:,:]**3 - ks[:,k0:-1,:]**3)
        dcf = torch.cat((0*dcf0, dcf), dim=1)
        dcf = torch.cat((torch.flip(dcf, dims=[1]), dcf), dim=1)
        
    dcf = dcf[0,:,[0]].to(torch.float32)

    return ktraj, dcf


def create_low_rank_subspace_basis(nechoes: int = 1000, ncoeff: int = 4) -> Tuple[Tensor, Tensor]:
    """ Generate low-rank temporal subspace basis for a Spin-Echo acquisition.

    Args:
        nechoes (int): number of echoes in the train.
        ncoeff (int): number of subspace coefficient to be retained.

    Returns:
        basis (tensor): Low-rank subspace basis.
        sig (tensor): Signal ensemble used to compute basis via SVD.
    """
    # assume T2 spin echo
    t2 = np.linspace(1, 329, 300)

    # create echos
    te = np.linspace(1, 300, nechoes)

    # simulate signals (analytically)
    sig = np.exp(-te[None, :] / t2[:, None])

    # get basis
    _, _, basis = np.linalg.svd(sig, full_matrices=False)

    # select subspace
    basis = basis[:ncoeff, :]

    return torch.tensor(basis, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32)  # pylint: disable=no-member


def create_shepp_logan_phantom(matrix_shape: Union[List[int], Tuple[int]],
                               nechoes: int = 1000,
                               ncoeff: int = 4) -> Tensor:
    """ Create low-rank subspace coefficients for a Shepp-Logan phantom.

        Assume Spin-Echo acquisition with infinite TR.

    Args:
        matrix_shape: Tuple or list of ints describing image FOV.
        nechoes (int): number of echoes in the train.
        ncoeff (int): number of subspace coefficient to be retained.

    Returns:
        tensor: low-rank subspace coefficient for the Shepp-Logan phantom.
    """
    # get tissue segmentation mask
    discrete_model = np.round(_shepp_logan(
        matrix_shape, dtype=np.float32)).astype(np.int32)

    # collapse vessels rois, csf rois and re-order indexes
    discrete_model[discrete_model == 1] = 1
    discrete_model[discrete_model == 2] = 1
    discrete_model[discrete_model == 3] = 1
    discrete_model[discrete_model == 4] = 1
    discrete_model[discrete_model == 5] = 2
    discrete_model[discrete_model == 6] = 2
    discrete_model[discrete_model == 7] = 3
    discrete_model[discrete_model == 8] = 3
    discrete_model = torch.tensor(discrete_model)  # pylint: disable=no-member

    # assign relaxation values to different regions values
    t2_wm = 70
    t2_gm = 83
    t2_csf = 329

    # collect in a single array
    t2 = np.array([t2_wm, t2_gm, t2_csf])

    # simulate
    te = np.linspace(1, 300, nechoes)
    sig = np.exp(-te[:, None] / t2[None, :])
    sig = torch.tensor(sig, dtype=torch.float32)  # pylint: disable=no-member

    # get basis
    basis, _ = create_low_rank_subspace_basis(nechoes, ncoeff)
    sig = basis @ sig

    # assign to tissue mask to create output image
    output = torch.zeros((ncoeff, *matrix_shape),  # pylint: disable=no-member
                         dtype=torch.float32)

    for n in range(ncoeff):
        output[n, discrete_model == 1] = sig[n, 0]
        output[n, discrete_model == 2] = sig[n, 1]
        output[n, discrete_model == 3] = sig[n, 2]

    if len(matrix_shape) == 2:
        output = output.unsqueeze(1)
        
    return output


def project_kspace_data(kdata: Tensor, basis: Tensor):
    """ Project kspace data to subspace for standard nufft batch computation.

    Args:
        kdata (tensor): input k-space data.
        basis (tensor): input subspace basis.

    Returns:
        tensor: subspace projected data.
    """
    ncoeff = basis.shape[0]

    # preallocate output
    output = torch.zeros((ncoeff, *kdata.shape),  # pylint: disable=no-member
                         dtype=kdata.dtype)

    # project over low-rank subspace
    for n in range(ncoeff):
        output[n] = (basis[n] * kdata.transpose()).transpose()

    return output


def plot_kspace_trajectory(ktraj: Tensor):
    """ Plot k-space coordinates.

    Args:
        ktraj (tensor): tensor of k-space coordinates.
    """
    # Get dimensions
    ndim = ktraj.shape[-1]

    if ndim == 2:
        print(ktraj.shape)
        plt.plot(ktraj[:40, :, 0, 0].T, ktraj[:40, :, 0, 1].T)
        plt.title('K-Space trajectory', fontsize=20)
        plt.xlabel("kx [a.u.]", fontsize=20)
        plt.ylabel("ky [a.u.]", fontsize=20)

    if ndim == 3:
        ax = plt.subplot(111, projection='3d')
        for n in range(40):
            ax.plot3D(ktraj[n, :, 0, 0].T, ktraj[n, :, 0, 1].T, ktraj[n, :, 0, 2].T)
        ax.set_title('K-Space trajectory', fontsize=20)
        ax.set_xlabel("kx [a.u.]", fontsize=20)
        ax.set_ylabel("ky [a.u.]", fontsize=20)
        ax.set_zlabel("kz [a.u.]", fontsize=20)


def plot_signal_and_basis(basis: Tensor, signal: Union[None, Tensor] = None):
    """ Plot svd basis and signal ensamble.

    Args:
        basis (tensor): tensor of subspace basis.
        signal (tensor): signal ensemble matrix.

    """
    nframes = basis.shape[-1]

    if signal is not None:
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(torch.arange(nframes), signal.T)
        axs[0].set_title('Signal ensemble', fontsize=20)
        axs[0].set_xlabel("# frames", fontsize=20)
        axs[0].set_ylabel("signal magnitude [a.u.]", fontsize=20)

        axs[1].plot(torch.arange(nframes), basis.T)
        axs[1].set_title('Low-rank subspace basis', fontsize=20)
        axs[1].set_xlabel("# frames", fontsize=20)
        axs[1].set_ylabel("signal magnitude [a.u.]", fontsize=20)

    else:
        plt.plot(nframes, basis.T)
        plt.title('Low-rank subspace basis', fontsize=20)
        plt.xlabel("# frames", fontsize=20)
        plt.ylabel("signal magnitude [a.u.]", fontsize=20)


def show_image_series(image_series: Union[List, Tuple, Tensor], slice_idx: Union[Tensor]):
    """ Show image series (i.e. set of singular values).

    Args:
        image_series: either a tensor of image series or a list/tuple of tensors.
    """
    if isinstance(image_series, list) or isinstance(image_series, tuple):
        x = []
        for vol in range(len(image_series)):
            tmp = [torch.flip(image_series[vol][n][slice_idx], dims=[-1]) / torch.max(  # pylint: disable=no-member
                torch.abs(image_series[vol][n][slice_idx])) for n in range(image_series[vol].shape[0])]  # pylint: disable=no-member
            tmp = torch.cat(tmp, dim=1)  # pylint: disable=no-member
            x.append(tmp)
        x = torch.cat(x, dim=0)  # pylint: disable=no-member

    else:
        x = [torch.flip(image_series[n][slice_idx], dims=[-1]) / torch.abs(image_series[n]  # pylint: disable=no-member
                                                                        [slice_idx]).max() for n in range(image_series.shape[0])]
        x = torch.cat(x, dim=1)  # pylint: disable=no-member

    # show
    plt.imshow(torch.abs(x), cmap='gray',
               interpolation='lanczos'), plt.axis('off')


# %% Utils


def _create_2d_radial(matrix_size, nframes):
    # create spoke
    spokelength = matrix_size * 2
    kx = np.linspace(-matrix_size // 2, matrix_size // 2, spokelength)
    ky = 0 * kx
    ktraj = np.stack((kx, ky), axis=0)[:, None, :]

    # rotate trajectory
    phi = np.deg2rad(_golden_angle_list(nframes))
    ktraj = _2d_rotation(ktraj, phi[:,None])

    return torch.tensor(ktraj.transpose())  # pylint: disable=no-member


def _golden_angle_list(length):
    golden_ratio = (np.sqrt(5.0) + 1.0) / 2.0
    conj_golden_ratio = 1 / golden_ratio

    m = np.arange(length, dtype=np.float32) * conj_golden_ratio
    phi = (180 * m) % 360

    return phi


def _2d_rotation(coord_in, phi):
    coord_out = np.zeros(
        (coord_in.shape[0], phi.shape[0], coord_in.shape[-1]), dtype=coord_in.dtype)
    coord_out[0] = coord_in[0] * np.cos(phi) - coord_in[1] * np.sin(phi)
    coord_out[1] = coord_in[0] * np.sin(phi) + coord_in[1] * np.cos(phi)

    return coord_out


def _create_3d_radial(matrix_size, nframes):
    # create spoke
    spokelength = matrix_size * 2
    kz = np.linspace(-matrix_size // 2, matrix_size // 2, spokelength)
    ky = 0 * kz
    kx = 0 * kz
    ktraj = np.stack((kx, ky, kz), axis=0)[:, None, :]

    # rotate trajectory
    phi, theta = _golden_means_list(nframes)
    phi, theta = np.deg2rad(phi), np.deg2rad(theta)

    ktraj = _3d_rotation(ktraj, phi[:,None], theta[:,None])

    return torch.tensor(ktraj.transpose())  # pylint: disable=no-member


def _golden_means_list(length):
    golden_mean_1 = 0.4656
    golden_mean_2 = 0.6823

    m1 = np.arange(length, dtype=np.float32) * golden_mean_1
    m2 = np.arange(length, dtype=np.float32) * golden_mean_2

    phi = (180 * m1) % 360
    theta = (np.arccos(m2 % 1) * 180 / np.pi) % 360

    return phi, theta


def _3d_rotation(coord_in, phi, theta):
    coord_out = np.zeros(
        (coord_in.shape[0], phi.shape[0], coord_in.shape[-1]), dtype=coord_in.dtype)
    coord_out[0] = coord_in[0] * (np.cos(theta) * np.cos(phi)) + coord_in[1] * \
        np.sin(phi) - coord_in[2] * (np.sin(theta) * np.cos(phi))
    coord_out[1] = -coord_in[0] * (np.cos(theta) * np.sin(phi)) + coord_in[1] * \
        np.cos(phi) + coord_in[2] * (np.sin(theta) * np.sin(phi))
    coord_out[2] = coord_in[0] * np.sin(theta) + coord_in[2] * np.cos(theta)

    return coord_out


def _prepare_trajectory_for_torchkbnufft(ktraj):

    # get distance from k-spoace center
    kabs = (ktraj**2).sum(axis=-1)**0.5

    # copy to avoid modification to original coord
    ktraj = ktraj.clone()

    # rescale to -pi:pi
    ktraj = ktraj / kabs.max() * np.pi
    
    # reshape
    ktraj = ktraj.reshape(np.prod(ktraj.shape[:-1]), ktraj.shape[-1])
    ktraj = ktraj.T

    return ktraj


def tkbnufft_factory(ktraj, im_size, oversamp=1.125, width=3):
    """Prepare NUFFT operator using torchkbnufft."""
    # adapt trajectory
    ktraj = _prepare_trajectory_for_torchkbnufft(ktraj)
    
    # get oversampled grid
    grid_size = tuple([int(el * oversamp) for el in im_size])
    forw_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, numpoints=width)
    
    def nufft(img):
        return forw_ob(img, ktraj)
    
    return nufft


def tkbnufft_adjoint_factory(ktraj, im_size, oversamp=1.125, width=3):
    """Prepare NUFFT Adjoint operator using torchkbnufft."""
    # adapt trajectory
    ktraj = _prepare_trajectory_for_torchkbnufft(ktraj)
    
    # get oversampled grid
    grid_size = tuple([int(el * oversamp) for el in im_size])
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, numpoints=width)
    
    def nufft_adjoint(img):
        return adj_ob(img, ktraj)
    
    return nufft_adjoint


def tkbnufft_selfadjoint_factory(ktraj, im_size, dcf):
    """Prepare NUFFT Selfadjoint operator using torchkbnufft."""
    # adapt trajectory
    ktraj = _prepare_trajectory_for_torchkbnufft(ktraj)
    
    # get oversampled grid
    dcomp_kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size, weights=dcf)  # with density compensation
    toep_ob = tkbn.ToepNufft()

    def nufft_selfadjoint(img):
        return toep_ob(img, dcomp_kernel)
    
    return nufft_selfadjoint


def _shepp_logan(shape, dtype=np.complex64):
    return _phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)


sl_amps = [8, 7, 6, 5, 4, 3, 2, 1]


sl_scales = [[.6900, .920, .810],  # white big
             [.6624, .874, .780],  # gray big
             [.1100, .310, .220],  # right black
             [.1600, .410, .280],  # left black
             [.2100, .250, .410],  # gray center blob
             [.0460, .046, .050],  # left small dot
             [.0230, .023, .020],  # mid small dot
             [.0230, .023, .020]]


def _phantom(shape, amps, scales, offsets, angles, dtype):
    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])
    elif len(shape) == 3:
        ndim = 3
    else:
        raise ValueError('Incorrect dimension')

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[-(shape[-3] // 2):((shape[-3] + 1) // 2),
                       -(shape[-2] // 2):((shape[-2] + 1) // 2),
                       -(shape[-1] // 2):((shape[-1] + 1) // 2)]

    coords = np.stack((x.ravel() / shape[-1] * 2,
                       y.ravel() / shape[-2] * 2,
                       z.ravel() / shape[-3] * 2))

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):
        _ellipsoid(amp, scale, offset, angle, coords, out)
    if ndim == 2:
        return out[0, :, :]
    else:
        return out


def _ellipsoid(amp, scale, offset, angle, coords, out):
    R = _rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / \
        np.reshape(scale, (3, 1))

    r2 = np.sum(coords ** 2, axis=0).reshape(out.shape)

    out[r2 <= 1] = amp


def _rotation_matrix(angle):
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
              -spsi * sphi + ctheta * cphi * cpsi,
              cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.array(alpha)


sl_offsets = [[0., 0., 0],
              [0., -.0184, 0],
              [.22, 0., 0],
              [-.22, 0., 0],
              [0., .35, -.15],
              [-.08, -.605, 0],
              [0., -.606, 0],
              [.06, -.605, 0]]

sl_angles = [[0, 0, 0],
             [0, 0, 0],
             [-18, 0, 10],
             [18, 0, 10],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]


