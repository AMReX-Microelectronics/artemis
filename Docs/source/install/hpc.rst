.. _install-hpc:

HPC
===

On selected high-performance computing (HPC) systems, Artemis has documented or even pre-build installation routines.
Follow the guide here instead of the generic installation routines for optimal stability and best performance.

artemis.profile
-------------

Use a ``artemis.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   source $HOME/artemis.profile

We list example ``artemis.profile`` files below, which can be used to set up Artemis on various HPC systems.

HPC Systems
-----------

.. toctree::
   :maxdepth: 1

   hpc/cori
   hpc/perlmutter
   hpc/summit
   hpc/spock
   hpc/crusher
   hpc/frontier
   hpc/fugaku
   hpc/juwels
   hpc/lassen
   hpc/quartz
   hpc/lawrencium
   hpc/ookami
   hpc/lxplus
   hpc/lumi
   hpc/taurus

.. tip::

   Your HPC system is not in the list?
   `Open an issue <https://github.com/ECP-WarpX/WarpX/issues>`__ and together we can document it!
