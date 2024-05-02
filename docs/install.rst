Installation
============

To install KMM first clone the repository, including the submodules, with the following command:

.. code-block:: shell

    git clone --recurse-submodules https://github.com/NLeSC-COMPAS/kmm.git

After cloning the repository on your machine, you can build KMM using ``cmake`` and ``make``:

.. code-block:: shell

    cd kmm/build
    cmake ..
    make

There are four CMake user options that it is possible to set:

* ``KMM_USE_CUDA``: to enable CUDA support in KMM
* ``KMM_BUILD_TESTS``: to build the tests
* ``KMM_BUILD_EXAMPLES``: to build the examples
* ``KMM_STATIC``: to build KMM as a static library
