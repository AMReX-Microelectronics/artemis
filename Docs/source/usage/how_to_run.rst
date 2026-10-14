.. _usage_run:

Run Artemis
=========

In order to run a new simulation:

#. create a **new directory**, where the simulation will be run
#. make sure the Artemis **executable** is either copied into this directory or in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__
#. add an **inputs file** and on :ref:`HPC systems <install-hpc>` a **submission script** to the directory
#. run

1. Run Directory
----------------

On Linux/macOS, this is as easy as this

.. code-block:: bash

   mkdir -p <run_directory>

Where ``<run_directory>`` by the actual path to the run directory.

2. Executable
-------------

If you installed warpX with a :ref:`package manager <install-users>`, a ``artemis``-prefixed executable will be available as a regular system command to you.
Depending on the choosen build options, the name is suffixed with more details.
Try it like this:

.. code-block:: bash

   artemis<TAB>

Hitting the ``<TAB>`` key will suggest available Artemis executables as found in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__.

.. note::

   Artemis needs separate binaries to run in dimensionality of 1D, 2D, 3D, and RZ.
   We encode the supported dimensionality in the binary file name.

If you :ref:`compiled the code yourself <install-developers>`, the Artemis executable is stored in the source folder under ``build/bin``.
We also create a symbolic link that is just called ``artemis`` that points to the last executable you built, which can be copied, too.
Copy the **executable** to this directory:

.. code-block:: bash

   cp build/bin/<artemis_executable> <run_directory>/

where ``<artemis_executable>`` should be replaced by the actual name of the executable (see above) and ``<run_directory>`` by the actual path to the run directory.

3. Inputs
---------

Add an **input file** in the directory (see :ref:`examples <usage-examples>` and :ref:`parameters <running-cpp-parameters>`).
This file contains the numerical and physical parameters that define the situation to be simulated.

On :ref:`HPC systems <install-hpc>`, also copy and adjust a submission script that allocated computing nodes for you.
Please :ref:`reach out to us <contact>` if you need help setting up a template that runs with ideal performance.

4. Run
------

**Run** the executable, e.g. with MPI:

.. code-block:: bash

   cd <run_directory>

   # run with an inputs file:
   mpirun -np <n_ranks> ./artemis <input_file>

or

.. code-block:: bash

   # run with a PICMI input script:
   mpirun -np <n_ranks> python <python_script>

Here, ``<n_ranks>`` is the number of MPI ranks used, and ``<input_file>`` is the name of the input file (``<python_script>`` is the name of the :ref:`PICMI <usage-picmi>` script).
Note that the actual executable might have a longer name, depending on build options.

We used the copied executable in the current directory (``./``); if you installed with a package manager, skip the ``./`` because Artemis is in your ``PATH``.

On an :ref:`HPC system <install-hpc>`, you would instead submit the :ref:`job script <install-hpc>` at this point, e.g. ``sbatch <submission_script>`` (SLURM on Cori/NERSC) or ``bsub <submission_script>`` (LSF on Summit/OLCF).

.. tip::

   In the :ref:`next sections <running-cpp-parameters>`, we will explain parameters of the ``<input_file>``.
   You can overwrite all parameters inside this file also from the command line, e.g.:

   .. code-block:: bash

      mpirun -np 4 ./artemis <input_file> max_step=10 artemis.numprocs=1 2 2

5. Outputs
----------

By default, Artemis will write a status update to the terminal (``stdout``).
On :ref:`HPC systems <install-hpc>`, we usually store a copy of this in a file called ``outputs.txt``.

We also store by default an exact copy of all explicitly and implicitly used inputs parameters in a file called ``artemis_used_inputs`` (this file name can be changed).
This is important for reproducibility, since as we wrote in the previous paragraph, the options in the input file can be extended and overwritten from the command line.

:ref:`Further configured diagnostics <running-cpp-parameters-diagnostics>` are explained in the next sections.
By default, they are written to a subdirectory in ``diags/`` and can use various :ref:`output formats <dataanalysis-formats>`.
