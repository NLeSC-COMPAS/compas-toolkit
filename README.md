<p align="center">
  <img src="https://avatars.githubusercontent.com/u/132893361" width=230 />
</p>


# Compas Toolkit

[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/NLeSC-COMPAS/compas-toolkit)
![CUDA Build](https://github.com/NLeSC-COMPAS/compas-toolkit/actions/workflows/cmake-cuda-multi-compiler.yml/badge.svg)
![HIP Build](https://github.com/NLeSC-COMPAS/compas-toolkit/actions/workflows/cmake-hip.yml/badge.svg)
![GitHub License](https://img.shields.io/github/license/NLeSC-COMPAS/compas-toolkit)
![GitHub Tag](https://img.shields.io/github/v/tag/NLeSC-COMPAS/compas-toolkit)

The Compas Toolkit is a high-performance C++ library offering GPU-accelerated functions for use in quantitative MRI research.
The toolkit offers fast simulations of various MRI sequences and k-space trajectories commonly used in qMRI studies.
While the core of the toolkit is implemented using CUDA, the functionality is accessible from both C++ and Julia.


## Features

* Flexible API that can be composed in different ways.
* Highly tuned GPU kernels that provide high performance.
* Implemented using CUDA, optimized for Nvidia GPUs.
* Usable from Julia and C++.


## Prerequisites

Compilation requires the following software:

- CMake (version 3.10 or higher)
- NVIDIA CUDA Compiler (version 11.0 or higher)
- Julia (version 1.9 or later, only for Julia bindings)


## Installation

To install the Compas Toolkit, follow these steps:

### Clone the repository

First, clone the GitHub repository:

```bash
$ git clone --recurse-submodules https://github.com/NLeSC-COMPAS/compas-toolkit
```

### Compiling the C++ code

Next, configure the CMake project inside a new `build` directory:

```bash
$ mkdir -p build
$ cd build
$ cmake -B. -DCOMPAS_USE_CUDA=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 ..
```

After the configuration, build the toolkit by running:

```bash
$ make compas-toolkit
```

This generates a static library named `libcompas-toolkit.a`.

### Compiling the Julia bindings

Please visit [CompasToolkit.jl](https://github.com/NLeSC-COMPAS/CompasToolkit.jl) for details on how to use Compas Toolkit from Julia.

