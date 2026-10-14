.. _install-developers:
.. _building-cmake:
.. _building-cmake-intro:

Developers
==========

`CMake <https://cmake.org>`_ is our primary build system.
If you are new to CMake, `this short tutorial <https://hsf-training.github.io/hsf-training-cmake-webpage/>`_ from the HEP Software foundation is the perfect place to get started.
If you just want to use CMake to build the project, jump into sections `1. Introduction <https://hsf-training.github.io/hsf-training-cmake-webpage/01-intro/index.html>`__, `2. Building with CMake <https://hsf-training.github.io/hsf-training-cmake-webpage/02-building/index.html>`__ and `9. Finding Packages <https://hsf-training.github.io/hsf-training-cmake-webpage/09-findingpackages/index.html>`__.


Dependencies
------------

Before you start, you will need a copy of the Artemis source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/artemis
   cd $HOME/src/artemis

Artemis depends on popular third party software.

* On your development machine, :ref:`follow the instructions here <install-dependencies>`.
* If you are on an HPC machine, :ref:`follow the instructions here <install-hpc>`.

.. toctree::
   :hidden:

   dependencies


Compile
-------

From the base of the Artemis source directory, execute:

.. code-block:: bash

   # find dependencies & configure
   #   see additional options below, e.g.
   #                   -DCMAKE_INSTALL_PREFIX=$HOME/sw/artemis
   cmake -S . -B build

   # compile, here we use four threads
   cmake --build build -j 4

**That's it!**
A 3D Artemis binary is now in ``build/bin/`` and :ref:`can be run <usage_run>` with a :ref:`3D example inputs file <usage-examples>`.
Most people execute the binary directly or copy it out.

If you want to install the executables in a programmatic way, run this:

.. code-block:: bash

   # for default install paths, you will need administrator rights, e.g. with sudo:
   cmake --build build --target install

You can inspect and modify build options after running ``cmake -S . -B build`` with either

.. code-block:: bash

   ccmake build

or by adding arguments with ``-D<OPTION>=<VALUE>`` to the first CMake call.
For example, this builds Artemis in 2D geometry and enables Nvidia GPU (CUDA) support:

.. code-block:: bash

   cmake -S . -B build -DArtemis_DIMS=2 -DArtemis_COMPUTE=CUDA


.. _building-cmake-options:

Build Options
-------------

============================= ============================================ =========================================================
CMake Option                  Default & Values                             Description
============================= ============================================ =========================================================
``CMAKE_BUILD_TYPE``          RelWithDebInfo/**Release**/Debug             Type of build, symbols & optimizations
``CMAKE_INSTALL_PREFIX``      system-dependent path                        Install path prefix
``CMAKE_VERBOSE_MAKEFILE``    ON/**OFF**                                   Print all compiler commands to the terminal during build
``PYINSTALLOPTIONS``                                                       Additional options for ``pip install``, e.g., ``-v --user``
``Artemis_APP``                 **ON**/OFF                                   Build the Artemis executable application
``Artemis_ASCENT``              ON/**OFF**                                   Ascent in situ visualization
``Artemis_COMPUTE``             NOACC/**OMP**/CUDA/SYCL/HIP                  On-node, accelerated computing backend
``Artemis_DIMS``                **3**/2/1/RZ                                 Simulation dimensionality
``Artemis_EB``                  ON/**OFF**                                   Embedded boundary support (not supported in RZ yet)
``Artemis_GPUCLOCK``            **ON**/OFF                                   Add GPU kernel timers (cost function, +4 registers/kernel)
``Artemis_IPO``                 ON/**OFF**                                   Compile Artemis with interprocedural optimization (aka LTO)
``Artemis_LIB``                 ON/**OFF**                                   Build Artemis as a shared library, e.g., for PICMI Python
``Artemis_MPI``                 **ON**/OFF                                   Multi-node support (message-passing)
``Artemis_MPI_THREAD_MULTIPLE`` **ON**/OFF                                   MPI thread-multiple support, i.e. for ``async_io``
``Artemis_OPENPMD``             **ON**/OFF                                   openPMD I/O (HDF5, ADIOS)
``Artemis_PRECISION``           SINGLE/**DOUBLE**                            Floating point precision (single/double)
``Artemis_PARTICLE_PRECISION``  SINGLE/**DOUBLE**                            Particle floating point precision (single/double), defaults to Artemis_PRECISION value if not set
``Artemis_PSATD``               ON/**OFF**                                   Spectral solver
``Artemis_QED``                 **ON**/OFF                                   QED support (requires PICSAR)
``Artemis_QED_TABLE_GEN``       ON/**OFF**                                   QED table generation support (requires PICSAR and Boost)
``Artemis_SENSEI``              ON/**OFF**                                   SENSEI in situ visualization
============================= ============================================ =========================================================

Artemis can be configured in further detail with options from AMReX, which are documented in the AMReX manual:

* `general AMReX build options <https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options>`__
* `GPU-specific options <https://amrex-codes.github.io/amrex/docs_html/GPU.html#building-gpu-support>`__.

**Developers** might be interested in additional options that control dependencies of Artemis.
By default, the most important dependencies of Artemis are automatically downloaded for convenience:

============================= ============================================== ===========================================================
CMake Option                  Default & Values                               Description
============================= ============================================== ===========================================================
``BUILD_SHARED_LIBS``         ON/**OFF**                                     Build shared libraries for dependencies
``CCACHE_PROGRAM``            First found ``ccache`` executable.             Set to ``-DCCACHE_PROGRAM=NO`` to disable CCache.
``AMReX_CUDA_PTX_VERBOSE``    ON/**OFF**                                     Print CUDA code generation statistics from ``ptxas``.
``Artemis_amrex_src``           *None*                                         Path to AMReX source directory (preferred if set)
``Artemis_amrex_repo``          ``https://github.com/AMReX-Codes/amrex.git``   Repository URI to pull and build AMReX from
``Artemis_amrex_branch``        *we set and maintain a compatible commit*      Repository branch for ``Artemis_amrex_repo``
``Artemis_amrex_internal``      **ON**/OFF                                     Needs a pre-installed AMReX library if set to ``OFF``
``Artemis_openpmd_src``         *None*                                         Path to openPMD-api source directory (preferred if set)
``Artemis_openpmd_repo``        ``https://github.com/openPMD/openPMD-api.git`` Repository URI to pull and build openPMD-api from
``Artemis_openpmd_branch``      ``0.15.1``                                     Repository branch for ``Artemis_openpmd_repo``
``Artemis_openpmd_internal``    **ON**/OFF                                     Needs a pre-installed openPMD-api library if set to ``OFF``
``Artemis_picsar_src``          *None*                                         Path to PICSAR source directory (preferred if set)
``Artemis_picsar_repo``         ``https://github.com/ECP-WarpX/picsar.git``    Repository URI to pull and build PICSAR from
``Artemis_picsar_branch``       *we set and maintain a compatible commit*      Repository branch for ``Artemis_picsar_repo``
``Artemis_picsar_internal``     **ON**/OFF                                     Needs a pre-installed PICSAR library if set to ``OFF``
============================= ============================================== ===========================================================

For example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in ``$HOME/src/amrex``, add the ``cmake`` argument ``-DArtemis_amrex_src=$HOME/src/amrex``.
Relative paths are also supported, e.g. ``-DArtemis_amrex_src=../amrex``.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to ``https://github.com/WeiqunZhang/amrex/`` in a branch ``new-feature`` then pass to ``cmake`` the arguments: ``-DArtemis_amrex_repo=https://github.com/WeiqunZhang/amrex.git -DArtemis_amrex_branch=new-feature``.

You can speed up the install further if you pre-install these dependencies, e.g. with a package manager.
Set ``-DArtemis_<dependency-name>_internal=OFF`` and add installation prefix of the dependency to the environment variable `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html>`__.
Please see the :ref:`introduction to CMake <building-cmake-intro>` if this sounds new to you.

If you re-compile often, consider installing the `Ninja <https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages>`__ build system.
Pass ``-G Ninja`` to the CMake configuration call to speed up parallel compiles.


.. _building-cmake-envvars:

Configure your compiler
-----------------------

If you don't want to use your default compiler, you can set the following environment variables.
For example, using a Clang/LLVM:

.. code-block:: bash

   export CC=$(which clang)
   export CXX=$(which clang++)

If you also want to select a CUDA compiler:

.. code-block:: bash

   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=$(which clang++)

We also support adding `additional compiler flags via environment variables <https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cmake-language-environment-variables>`__ such as `CXXFLAGS <https://cmake.org/cmake/help/latest/envvar/CXXFLAGS.html>`__/`LDFLAGS <https://cmake.org/cmake/help/latest/envvar/LDFLAGS.html>`__:

.. code-block:: bash

   # example: treat all compiler warnings as errors
   export CXXFLAGS="-Werror"

.. note::

   Please clean your build directory with ``rm -rf build/`` after changing the compiler.
   Now call ``cmake -S . -B build`` (+ further options) again to re-initialize the build configuration.


Run
---

An executable Artemis binary with the current compile-time options encoded in its file name will be created in ``build/bin/``.
Note that you need separate binaries to run 1D, 2D, 3D, and RZ geometry inputs scripts.
Additionally, a `symbolic link <https://en.wikipedia.org/wiki/Symbolic_link>`__ named ``artemis`` can be found in that directory, which points to the last built Artemis executable.

More details on running simulations are in the section :ref:`Run Artemis <usage_run>`.
Alternatively, read on and also build our PICMI Python interface.


.. _building-cmake-python:

PICMI Python Bindings
---------------------

.. note::

   Preparation: make sure you work with up-to-date Python tooling.

   .. code-block:: bash

      python3 -m pip install -U pip setuptools wheel
      python3 -m pip install -U cmake

For PICMI Python bindings, configure Artemis to produce a library and call our ``pip_install`` *CMake target*:

.. code-block:: bash

   # find dependencies & configure
   cmake -S . -B build -DArtemis_LIB=ON

   # build and then call "python3 -m pip install ..."
   cmake --build build --target pip_install -j 4

**That's it!**
You can now :ref:`run a first 3D PICMI script <usage-picmi>` from our :ref:`examples <usage-examples>`.

Developers could now change the Artemis source code and then call the build line again to refresh the Python installation.

.. note::

   These commands build one ``-DArtemis_DIMS=...`` dimensionality (default: ``3``) at a time.
   If your ``build/lib*/`` directory contains previously built ``libartemis*`` libraries, then ``--target pip_install`` picks them up as well.
   A new call to ``cmake --build build ...`` will only rebuild *one* dimensionality, as set via ``Artemis_DIMS``.

If you like to build a Artemis Python package that supports *all* dimensionalities, you can run this:

.. code-block:: bash

   for d in 1 2 3 RZ; do
       cmake -S . -B build -DArtemis_DIMS=$d -DArtemis_LIB=ON
       cmake --build build -j 4
   done
   cmake --build build --target pip_install

.. tip::

   If you do *not* develop with :ref:`a user-level package manager <install-dependencies>`, e.g., because you rely on a HPC system's environment modules, then consider to set up a virtual environment via `Python venv <https://docs.python.org/3/library/venv.html>`__.
   Otherwise, without a virtual environment, you likely need to add the CMake option ``-DPYINSTALLOPTIONS="--user"``.


.. _building-pip-python:

Python Bindings (Package Management)
------------------------------------

This section is relevant for Python package management, mainly for maintainers or people that rather like to interact only with ``pip``.

One can build and install ``pyartemis`` from the root of the Artemis source tree:

.. code-block:: bash

   python3 -m pip wheel -v .
   python3 -m pip install pyartemis*whl

This will call the CMake logic above implicitly.
Using this workflow has the advantage that it can build and package up multiple libraries with varying ``Artemis_DIMS`` into one ``pyartemis`` package.

Environment variables can be used to control the build step:

============================= ============================================ ================================================================
Environment Variable          Default & Values                             Description
============================= ============================================ ================================================================
``ARTEMIS_COMPUTE``             NOACC/**OMP**/CUDA/SYCL/HIP                  On-node, accelerated computing backend
``ARTEMIS_DIMS``                ``"1;2;3;RZ"``                               Simulation dimensionalities (semicolon-separated list)
``ARTEMIS_EB``                  ON/**OFF**                                   Embedded boundary support (not supported in RZ yet)
``ARTEMIS_MPI``                 ON/**OFF**                                   Multi-node support (message-passing)
``ARTEMIS_OPENPMD``             **ON**/OFF                                   openPMD I/O (HDF5, ADIOS)
``ARTEMIS_PRECISION``           SINGLE/**DOUBLE**                            Floating point precision (single/double)
``ARTEMIS_PARTICLE_PRECISION``  SINGLE/**DOUBLE**                            Particle floating point precision (single/double), defaults to Artemis_PRECISION value if not set
``ARTEMIS_PSATD``               ON/**OFF**                                   Spectral solver
``ARTEMIS_QED``                 **ON**/OFF                                   PICSAR QED (requires PICSAR)
``ARTEMIS_QED_TABLE_GEN``       ON/**OFF**                                   QED table generation (requires PICSAR and Boost)
``BUILD_PARALLEL``            ``2``                                        Number of threads to use for parallel builds
``BUILD_SHARED_LIBS``         ON/**OFF**                                   Build shared libraries for dependencies
``HDF5_USE_STATIC_LIBRARIES`` ON/**OFF**                                   Prefer static libraries for HDF5 dependency (openPMD)
``ADIOS_USE_STATIC_LIBS``     ON/**OFF**                                   Prefer static libraries for ADIOS1 dependency (openPMD)
``ARTEMIS_AMREX_SRC``           *None*                                       Absolute path to AMReX source directory (preferred if set)
``ARTEMIS_AMREX_REPO``          *None (uses cmake default)*                  Repository URI to pull and build AMReX from
``ARTEMIS_AMREX_BRANCH``        *None (uses cmake default)*                  Repository branch for ``ARTEMIS_AMREX_REPO``
``ARTEMIS_AMREX_INTERNAL``      **ON**/OFF                                   Needs a pre-installed AMReX library if set to ``OFF``
``ARTEMIS_OPENPMD_SRC``         *None*                                       Absolute path to openPMD-api source directory (preferred if set)
``ARTEMIS_OPENPMD_INTERNAL``    **ON**/OFF                                   Needs a pre-installed openPMD-api library if set to ``OFF``
``ARTEMIS_PICSAR_SRC``          *None*                                       Absolute path to PICSAR source directory (preferred if set)
``ARTEMIS_PICSAR_INTERNAL``     **ON**/OFF                                   Needs a pre-installed PICSAR library if set to ``OFF``
``ARTEMIS_CCACHE_PROGRAM``      First found ``ccache`` executable.           Set to ``NO`` to disable CCache.
``PYARTEMIS_LIB_DIR``           *None*                                       If set, search for pre-built Artemis C++ libraries (see below)
============================= ============================================ ================================================================

Note that we currently change the ``ARTEMIS_MPI`` default intentionally to ``OFF``, to simplify a first install from source.

Some hints and workflows follow.
Developers, that want to test a change of the source code but did not change the ``pyartemis`` version number, can force a reinstall via:

.. code-block:: bash

   python3 -m pip install --force-reinstall --no-deps -v .

Some Developers like to code directly against a local copy of AMReX, changing both code-bases at a time:

.. code-block:: bash

   ARTEMIS_AMREX_SRC=$PWD/../amrex python3 -m pip install --force-reinstall --no-deps -v .

Additional environment control as common for CMake (:ref:`see above <building-cmake-intro>`) can be set as well, e.g. ``CC``, `CXX``, and ``CMAKE_PREFIX_PATH`` hints.
So another sophisticated example might be: use Clang as the compiler, build with local source copies of PICSAR and AMReX, support the PSATD solver, MPI and openPMD, hint a parallel HDF5 installation in ``$HOME/sw/hdf5-parallel-1.10.4``, and only build 3D geometry:

.. code-block:: bash

   CC=$(which clang) CXX=$(which clang++) ARTEMIS_AMREX_SRC=$PWD/../amrex ARTEMIS_PICSAR_SRC=$PWD/../picsar ARTEMIS_PSATD=ON ARTEMIS_MPI=ON ARTEMIS_DIMS=3 CMAKE_PREFIX_PATH=$HOME/sw/hdf5-parallel-1.10.4:$CMAKE_PREFIX_PATH python3 -m pip install --force-reinstall --no-deps -v .

Here we wrote this all in one line, but one can also set all environment variables in a development environment and keep the pip call nice and short as in the beginning.
Note that you need to use absolute paths for external source trees, because pip builds in a temporary directory, e.g. ``export ARTEMIS_AMREX_SRC=$HOME/src/amrex``.

The Python library ``pyartemis`` can also be created by pre-building Artemis into one or more shared libraries externally.
For example, a package manager might split Artemis into a C++ package and a Python package.
If the C++ libraries are already pre-compiled, we can pick them up in the Python build step instead of compiling them again:

.. code-block:: bash

   # build Artemis executables and libraries
   for d in 1 2 3 RZ; do
     cmake -S . -B build -DArtemis_DIMS=$d -DArtemis_LIB=ON
     cmake --build build -j 4
   done

   # Python package
   PYARTEMIS_LIB_DIR=$PWD/build/lib python3 -m pip wheel .

   # install
   python3 -m pip install pyartemis-*whl


Artemis release managers might also want to generate a self-contained source package that can be distributed to exotic architectures:

.. code-block:: bash

   python setup.py sdist --dist-dir .
   python3 -m pip wheel -v pyartemis-*.tar.gz
   python3 -m pip install *whl

The above steps can also be executed in one go to build from source on a machine:

.. code-block:: bash

   python3 setup.py sdist --dist-dir .
   python3 -m pip install -v pyartemis-*.tar.gz

Last but not least, you can uninstall ``pyartemis`` as usual with:

.. code-block:: bash

   python3 -m pip uninstall pyartemis
