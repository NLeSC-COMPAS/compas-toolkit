Installation
============

Prerequisites
-------------

Compilation requires the following software:

- CMake (version 3.10 or higher)
- NVIDIA CUDA Compiler (version 11.0 or higher)
- Julia (version 1.9 or later, only for Julia bindings)


Clone the repository
--------------------

First, clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/NLeSC-COMPAS/compas-toolkit


Compiling the C++ code
----------------------

Next, configure the CMake project inside a new ``build`` directory:

.. code-block:: bash

    mkdir -p build
    cd build
    cmake -B. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 ..


After the configuration, build the toolkit by running:

.. code-block:: bash

    make compas-toolkit

This generates a static library named ``libcompas-toolkit.a``.

Compiling the Julia bindings
----------------------------

To compile the Julia bindings, use the following command.
This will install the library into a local ``lib/`` directory:

.. code-block:: bash

    make install


If everything has gone as planned, you'll now have a shared library called ``libcompas-julia.so`` in the local ``lib/`` directory. Additionally, a Julia file should have been automatically generated as ``CompasToolkit.jl/src/CompasToolkit.jl``.
