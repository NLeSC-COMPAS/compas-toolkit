# Compas Toolkit

The Compas Toolkit is a C++ library designed for use in quantitative MRI research, with GPU acceleration.
This toolkit is primarily implemented in CUDA, but it also offers bindings to access its functionality from the Julia programming language.


## Installation

Compilation requires the following software:

- CMake (version 3.10 or higher)
- NVIDIA CUDA Compiler (version 11.0 or higher)
- Julia (version 1.9 or later, only for Julia bindings)

### Compiling the C++ code

First, configure the CMake project inside a new `build` directory:

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

Compile the Julia bindings with the following command, which will install the library into a local `lib/` directory:

```bash
$ make install
```

If everything has gone as planned, you'll now have a shared library called `libcompas-julia.so` in the local `lib/` directory. Additionally, a Julia file should have been automatically generated as `CompasToolkit.jl/src/CompasToolkit.jl`.

## Usage

To use the Compas Toolkit in your Julia package, simply add the `CompasToolkit.jl` directory to your project dependencies. You can then import the library with `using CompasToolkit`.

For examples of using the toolkit, take a look at the scripts available in the `CompasToolkit.jl/tests` directory.
