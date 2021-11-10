# -*- coding: utf-8 -*-
""" Setup script for lr-nufft-torch."""
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

install_requires = ["torch>=1.10", "numpy", "numba"]

setuptools.setup(
    name="lr-nufft-torch",
    version=0.1,

    description="Pytorch-based NUFFT with embedded low-rank subspace projection.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/FiRMLAB-Pisa/lr-nufft-torch",

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

    packages=["lr-torch-nufft"],
    package_dir={"lrtorchnufft": "src"},
    python_requires=">=3.6",

    install_requires=["numpy"],
)
