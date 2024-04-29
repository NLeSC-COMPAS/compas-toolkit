# Compas Toolkit

[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/NLeSC-COMPAS/compas-toolkit)
![Github Build](https://github.com/NLeSC-COMPAS/compas-toolkit/actions/workflows/cmake-cuda-multi-compiler.yml/badge.svg)
![GitHub License](https://img.shields.io/github/license/NLeSC-COMPAS/compas-toolkit)
![GitHub Tag](https://img.shields.io/github/v/tag/NLeSC-COMPAS/compas-toolkit)

The Compas Toolkit is a high-performance C++ library offering GPU-accelerated kernels for functions frequently used in advanced quantitative MRI research.
It facilitates detailed simulations of various MRI sequences and k-space trajectories, which are critical in quantitative MRI studies.
While the core of the toolkit is implemented using CUDA, its features are accessible from both C++ and the Julia programming language.


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
$ git clone https://github.com/NLeSC-COMPAS/compas-toolkit
```

### Compiling the C++ code

Next, configure the CMake project inside a new `build` directory:

```bash
$ mkdir -p build
$ cd build
$ cmake -B. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 ..
```

After the configuration, build the toolkit by running:

```bash
$ make compas-toolkit
```

This generates a static library named `libcompas-toolkit.a`.

### Compiling the Julia bindings

To compile the Julia bindings, use the following command.
This will install the library into a local `lib/` directory:

```bash
$ make install
```

If everything has gone as planned, you'll now have a shared library called `libcompas-julia.so` in the local `lib/` directory. Additionally, a Julia file should have been automatically generated as `CompasToolkit.jl/src/CompasToolkit.jl`.

## Usage

To use the Compas Toolkit in Julia, simply build the Julia bindings using the instructions above and then add the directory `CompasToolkit.jl` to you Julia project.

```julia
$ julia
> using Pkg; Pkg.add(path="<path to compas-toolkit>/CompasToolkit.jl/")
```

You can then import the library with `using CompasToolkit`.

For examples of using the toolkit, take a look at the scripts available in the `CompasToolkit.jl/tests` directory.
