# -*- coding: utf-8 -*-
""" Setup script for nufft-torch."""
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

install_requires = ["torch>=1.10", "numpy", "numba", "pytest"]

setuptools.setup(
    name="nufftorch",
    version=0.5,

    description="Pytorch-based Fourier Transform with embedded low-rank subspace projection.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/FiRMLAB-Pisa/nufft-torch",

    author="Matteo Cencini",
    author_email="matteo.cencini@gmail.com",

    license="MIT",

    classifiers=[
        "Development Status ::2 - Pre-Alpha",

        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",

        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    keywords=["MRI", "model-based", "NUFFT"],

    packages=["nufftorch"],
    package_dir={"nufftorch": "nufftorch"},
    python_requires=">=3.6",

    install_requires=install_requires,
)
