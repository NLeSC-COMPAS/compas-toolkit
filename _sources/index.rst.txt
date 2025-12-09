.. highlight:: c++
   :linenothreshold: 1

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents

   Compas Toolkit <self>
   install
   example
   api_cxx
   Julia API <https://nlesc-compas.github.io/CompasToolkit.jl>
   Github Repository <https://github.com/NLeSC-COMPAS/compas-toolkit>

Compas Toolkit
===============

The Compas Toolkit is a high-performance C++ library offering GPU-accelerated functions for use in quantitative MRI research.
The toolkit offers fast simulations of various MRI sequences and k-space trajectories commonly used in qMRI studies.
While the core of the toolkit is implemented using CUDA, the functionality is accessible from both C++ and Julia.


Features
--------

* Flexible API that can be composed in different ways.
* Highly tuned GPU kernels that provide high performance.
* Implemented using CUDA, optimized for Nvidia GPUs.
* Usable from Julia and C++.

Usage
-----

To use the Compas Toolkit in Julia, simply build the Julia bindings using the instructions above and then add the directory ``CompasToolkit.jl`` to you Julia project.

.. code-block:: julia

   using Pkg; Pkg.add(path="<path to compas-toolkit>/CompasToolkit.jl/")


You can then import the library with ``using CompasToolkit``.

For examples of using the toolkit, take a look at the scripts available in the ``CompasToolkit.jl/tests`` directory.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

