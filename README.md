# NUFFT-Torch
Python-based differentiable NUFFT package optimized for multi-echo and model-based MRI.

## Getting Started

### Installation

NUFFT-Torch can be installed using pip:

1. Clone the repo

  ```
git clone git@github.com:FiRMLAB-Pisa/nufft-torch.git
  ```

2. Navigate to the repository root folder and install using pip

  ```
   pip install -e .
  ```

### Usage

Check `basic_example.py` in the `benchmark`folder for a simple usage example (Coming Soon: proper documentation and examples.)

## Other Packages

The package is inspired by:

1. [torchkbnufft](https://github.com/mmuckley/torchkbnufft) (Pytorch implementation of Michigan Image Reconstruction Toolbox NUFFT)
2. [SigPy](https://github.com/mikgroup/sigpy) (for Numpy arrays, Numba (for CPU) and CuPy (for GPU) backends)
3. [PyNUFFT](https://github.com/jyhmiinlin/pynufft) (for Numpy, also has PyCUDA/PyOpenCL backends)

