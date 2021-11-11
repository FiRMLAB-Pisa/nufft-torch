# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 23:19:42 2021

@author: admin
"""
from lr_nufft_torch import _test_utils as utils
from lr_nufft_torch import module

from lr_nufft_torch._util import prod

import torch

import time

import matplotlib.pyplot as plt
import numpy as np

def _setup_problem(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    if ndim == 2:
        shape = (npix, npix)
    else:
        shape = (npix, npix, npix)
    
    # generate data
    ground_truth = utils.create_shepp_logan_phantom(shape)
    
    # generate trajectory
    ktraj, dcf = utils.create_radial_trajectory(ndim, npix, nreadouts, nechoes)
    
    # generate low rank basis
    basis, _ = utils.create_low_rank_subspace_basis(nechoes=nechoes)
    
    return ground_truth, basis, ktraj, dcf, shape

def benchmark_lrnufftorch(npix=200, ndim=2, nechoes=1000, nreadouts=100000):
    
    # setup
    ground_truth, basis, ktraj, dcf, shape = _setup_problem(npix, ndim, nechoes, nreadouts)
    
    # generate k-space data
    F = module.NUFFT(ktraj, shape=shape, basis=basis)
    t0 = time.time()
    kdata = F(ground_truth.clone())
    time_fwd = time.time() - t0
        
    # reconstruct image using lr_nufft_torch
    FH = F.H
    t0 = time.time()
    image = FH(dcf * kdata.clone())
    time_adj = time.time() - t0

    # do self-adjoint
    # FHF = module.NUFFTSelfadjoint(ktraj, shape=shape, basis=basis, dcf=dcf)
    # time_selfadj = %timeit -o FHF(image.clone())
    
    return time_fwd, time_adj#, time_selfadj

def benchmark_torchkbnufft(npix=200, ndim=2, nechoes=1000, nreadouts=100000):   
    # setup
    ground_truth, basis, ktraj, dcf, shape = _setup_problem(npix, ndim, nechoes, nreadouts)
    
    # generate k-space data
    tkbF = utils.tkbnufft_factory(ktraj, im_size=shape)
    t0 = time.time()
    kdata = tkbF(ground_truth.clone().to(torch.complex64)).squeeze()
    time_fwd = time.time() - t0
    kdata = kdata.reshape((kdata.shape[0], *ktraj.shape[:-1]))  
    kdata = (basis[:,:,None,None] * kdata).sum(axis=0)
        
    # reconstruct using torchkbnufft
    tkbFH = utils.tkbnufft_adjoint_factory(ktraj, im_size=shape)
    kdata = basis[:,:,None,None] * (dcf * kdata[None,...])
    kdata = kdata.reshape((kdata.shape[0], 1, prod(kdata.shape[1:]))) 
    t0 = time.time()
    image = tkbFH(kdata.clone())
    time_adj = time.time() - t0

    # do-selfadjoint
    # tkbFHF = utils.tkbnufft_selfadjoint_factory(ktraj, im_size=shape, dcf=dcf)
    # time_selfadj = %timeit -o tkbFHF(image_tkb.clone())
    
    return time_fwd, time_adj#, time_selfadj

time_fwd_lrnftorch, time_adj_lrnftorch = benchmark_lrnufftorch() 
time_fwd_torchkbnufft, time_adj_torchkbnufft = benchmark_torchkbnufft()

labels = ['forward NUFFT', 'adjoint NUFFT']
time_tkb = [time_fwd_torchkbnufft, time_adj_torchkbnufft]
time_lr = [time_fwd_lrnftorch, time_adj_lrnftorch]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, time_tkb, width, label='torch-kb-nufft')
rects2 = ax.bar(x - width/2, time_lr, width, label='lr-nufft-torch')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Execution Time [s]', fontsize=20)
ax.set_title('Benchmark', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=20)

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)

fig.tight_layout()

plt.show()

