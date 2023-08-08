<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/deepmr.svg?branch=main)](https://cirrus-ci.com/github/<USER>/deepmr)
[![ReadTheDocs](https://readthedocs.org/projects/deepmr/badge/?version=latest)](https://deepmr.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/deepmr/main.svg)](https://coveralls.io/r/<USER>/deepmr)
[![PyPI-Server](https://img.shields.io/pypi/v/deepmr.svg)](https://pypi.org/project/deepmr/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/deepmr.svg)](https://anaconda.org/conda-forge/deepmr)
[![Monthly Downloads](https://pepy.tech/badge/deepmr/month)](https://pepy.tech/project/deepmr)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/deepmr)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

<picture>
    <source srcset="https://github.com/FiRMLAB-Pisa/nufft-torch/blob/refactor/docs/source/_static/nufftorch_logo_dark.png"  media="(prefers-color-scheme: dark)">
    <img src="https://github.com/FiRMLAB-Pisa/nufft-torch/blob/refactor/docs/source/_static/nufftorch_logo.png">
</picture>

Python-based differentiable NUFFT package optimized for multi-echo and model-based MRI.

## Getting Started

### Installation

NUFFT-Torch can be installed using pip:


1. Clone the repo

  ```
git clone git@github.com:FiRMLAB-Pisa/nufft-torch.git
  ```

2. Navigate to the repository root folder and install using pip:

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


<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
