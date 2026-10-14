.. _usage-picmi:

Python (PICMI)
==============

Artemis uses the `PICMI standard <https://github.com/picmi-standard/picmi>`__ for its Python input files.
Python version 3.7 or newer is required.

Example input files can be found in :ref:`the examples section <usage-examples>`.
In the input file, instances of classes are created defining the various aspects of the simulation.
The `Simulation` object is the central object, where the instances are passed,
defining the simulation time, field solver, registered species, etc.

.. _usage-picmi-parameters:

Classes
----------

Simulation and grid setup
^^^^^^^^^^^^^^^^^^^^^^^^^

Simulation
""""""""""
.. autoclass:: pyartemis.picmi.Simulation
    :members: step, add_species, add_laser, write_input_file

Constants
"""""""""
For convenience, the PICMI interface defines the following constants,
which can be used directly inside any PICMI script. The values are in SI units.

- ``picmi.constants.c``: The speed of light in vacuum.
- ``picmi.constants.ep0``: The vacuum permittivity :math:`\epsilon_0`
- ``picmi.constants.mu0``: The vacuum permeability :math:`\mu_0`
- ``picmi.constants.q_e``: The elementary charge (absolute value of the charge of an electron).
- ``picmi.constants.m_e``: The electron mass
- ``picmi.constants.m_p``: The proton mass

Field solvers define the updates of electric and magnetic fields.

ElectromagneticSolver
"""""""""""""""""""""
.. autoclass:: pyartemis.picmi.ElectromagneticSolver

ElectrostaticSolver
"""""""""""""""""""
.. autoclass:: pyartemis.picmi.ElectrostaticSolver

Cartesian3DGrid
"""""""""""""""
.. autoclass:: pyartemis.picmi.Cartesian3DGrid

Cartesian2DGrid
"""""""""""""""
.. autoclass:: pyartemis.picmi.Cartesian2DGrid

Cartesian1DGrid
"""""""""""""""
.. autoclass:: pyartemis.picmi.Cartesian1DGrid

CylindricalGrid
"""""""""""""""
.. autoclass:: pyartemis.picmi.CylindricalGrid

EmbeddedBoundary
""""""""""""""""
.. autoclass:: pyartemis.picmi.EmbeddedBoundary

Applied fields
^^^^^^^^^^^^^^

ConstantAppliedField
""""""""""""""""""""
.. autoclass:: pyartemis.picmi.ConstantAppliedField

AnalyticAppliedField
""""""""""""""""""""
.. autoclass:: pyartemis.picmi.AnalyticAppliedField

PlasmaLens
""""""""""
.. autoclass:: pyartemis.picmi.PlasmaLens

Mirror
""""""
.. autoclass:: pyartemis.picmi.Mirror

Diagnostics
^^^^^^^^^^^

ParticleDiagnostic
""""""""""""""""""
.. autoclass:: pyartemis.picmi.ParticleDiagnostic

FieldDiagnostic
"""""""""""""""
.. autoclass:: pyartemis.picmi.FieldDiagnostic

ElectrostaticFieldDiagnostic
""""""""""""""""""""""""""""
.. autoclass:: pyartemis.picmi.ElectrostaticFieldDiagnostic

Lab-frame diagnostics diagnostics are used when running boosted-frame simulations.

LabFrameFieldDiagnostic
"""""""""""""""""""""""
.. autoclass:: pyartemis.picmi.LabFrameFieldDiagnostic

Checkpoint
""""""""""
.. autoclass:: pyartemis.picmi.Checkpoint

Particles
^^^^^^^^^

Species objects are a collection of particles with similar properties.
For instance, background plasma electrons, background plasma ions and an externally injected beam could each be their own particle species.

Species
"""""""
.. autoclass:: pyartemis.picmi.Species

MultiSpecies
""""""""""""
.. autoclass:: pyartemis.picmi.MultiSpecies

Particle distributions can be used for to initialize particles in a particle species.

GaussianBunchDistribution
"""""""""""""""""""""""""
.. autoclass:: pyartemis.picmi.GaussianBunchDistribution

UniformDistribution
"""""""""""""""""""
.. autoclass:: pyartemis.picmi.UniformDistribution

AnalyticDistribution
""""""""""""""""""""
.. autoclass:: pyartemis.picmi.AnalyticDistribution

ParticleListDistribution
""""""""""""""""""""""""
.. autoclass:: pyartemis.picmi.ParticleListDistribution

Particle layouts determine how to microscopically place macro particles in a grid cell.

GriddedLayout
"""""""""""""
.. autoclass:: pyartemis.picmi.GriddedLayout

PseudoRandomLayout
""""""""""""""""""
.. autoclass:: pyartemis.picmi.PseudoRandomLayout

Other operations related to particles

CoulombCollisions
"""""""""""""""""
.. autoclass:: pyartemis.picmi.CoulombCollisions

MCCCollisions
"""""""""""""
.. autoclass:: pyartemis.picmi.MCCCollisions

Lasers
^^^^^^

Laser profiles can be used to initialize laser pulses in the simulation.

GaussianLaser
"""""""""""""
.. autoclass:: pyartemis.picmi.GaussianLaser

AnalyticLaser
"""""""""""""
.. autoclass:: pyartemis.picmi.AnalyticLaser

Laser injectors control where to initialize laser pulses on the simulation grid.

LaserAntenna
""""""""""""
.. autoclass:: pyartemis.picmi.LaserAntenna


.. _usage-picmi-run:

Running
-------

Artemis can be run in one of two modes. It can run as a preprocessor, using the
Python input file to generate an input file to be used by the C++ version, or
it can be run directly from Python.
The examples support running in both modes by commenting and uncommenting the appropriate lines.

In either mode, if using a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`__, be sure to activate it before compiling and running Artemis.


Running Artemis directly from Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this, a full Python installation of Artemis is required, as described in :ref:`the install documentation <install-users>` (:ref:`developers <install-developers>`).

In order to run a new simulation:

* Create a **new directory**, where the simulation will be run.

* Add a **Python script** in the directory.

The input file should have the line ``sim.step()`` which runs the simulation.

* **Run** the script with Python:

.. code-block:: bash

   mpirun -np <n_ranks> python <python_script>

where ``<n_ranks>`` is the number of MPI ranks used, and ``<python_script>``
is the name of the script.


Extending a Simulation from Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running Artemis directly from Python it is possible to interact with the simulation
by installing ``CallbackFunctions``, which will execute a given Python function at a
specific location in the Artemis simulation loop.

.. autoclass:: pyartemis.callbacks.CallbackFunctions

Places in the Artemis loop where callbacks are available include:
``afterinit``, ``beforecollisions``, ``aftercollisions``, ``beforeEsolve``, ``afterEsolve``,
``beforeInitEsolve``, ``afterInitEsolve``, ``beforedeposition``, ``afterdeposition``,
``beforestep``, ``afterstep``, ``afterdiagnostics``,``afterrestart`` and ``oncheckpointsignal``.
See the examples in *Examples/Tests/ParticleDataPython* for references on how to use
``callbacks``.

There are several "hooks" available via the ``libartemis`` shared library to access and manipulate
simulation objects (particles, fields and memory buffers) as well as general properties
(such as processor number). These "hooks" are accessible through the `Simulation.extension` object.

.. autofunction:: pyartemis.picmi.Simulation.extension.getNProcs

.. autofunction:: pyartemis.picmi.Simulation.extension.getMyProc

.. autofunction:: pyartemis.picmi.Simulation.extension.get_nattr

.. autofunction:: pyartemis.picmi.Simulation.extension.get_nattr_species

.. autofunction:: pyartemis.picmi.Simulation.extension.getistep

.. autofunction:: pyartemis.picmi.Simulation.extension.gett_new

.. autofunction:: pyartemis.picmi.Simulation.extension.evolve

.. autofunction:: pyartemis.picmi.Simulation.extension.finalize

.. autofunction:: pyartemis.picmi.Simulation.extension.getProbLo

.. autofunction:: pyartemis.picmi.Simulation.extension.getProbHi

.. autofunction:: pyartemis.picmi.Simulation.extension.getCellSize

Particles can be added to the simulation at specific positions and with specific
attribute values:

.. autofunction:: pyartemis.picmi.Simulation.extension.add_particles

Properties of the particles already in the simulation can be obtained with various
functions.

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_count

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_structs

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_arrays

The ``get_particle_structs()`` and ``get_particle_arrays()`` functions are called
by several utility functions of the form ``get_particle_{comp_name}`` where
``comp_name`` is one of ``x``, ``y``, ``z``, ``r``, ``theta``, ``id``, ``cpu``,
``weight``, ``ux``, ``uy`` or ``uz``.

The index of some specific component of the particle data can be obtained.

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_comp_index

New components can be added via Python.

.. autofunction:: pyartemis.picmi.Simulation.extension.add_real_comp

Various diagnostics are also accessible from Python.
This includes getting the deposited or total charge density from a given species
as well as accessing the scraped particle buffer. See the example in
*Examples/Tests/ParticleBoudaryScrape* for a reference on how to interact
with scraped particle data.

.. autofunction:: pyartemis.picmi.Simulation.extension.get_species_charge_sum

.. autofunction:: pyartemis.picmi.Simulation.extension.depositChargeDensity

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_boundary_buffer_size

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_boundary_buffer_structs

.. autofunction:: pyartemis.picmi.Simulation.extension.get_particle_boundary_buffer

.. autofunction:: pyartemis.picmi.Simulation.extension.clearParticleBoundaryBuffer

The embedded boundary conditions can be modified when using the electrostatic solver.

.. autofunction:: pyartemis.picmi.Simulation.extension.set_potential_EB

Using Python input as a preprocessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, only the pure Python version needs to be installed, as described :ref:`here <developers-gnumake-python>`.

In order to run a new simulation:

* Create a **new directory**, where the simulation will be run.

* Add a **Python script** in the directory.

The input file should have the line like ``sim.write_input_file(file_name = 'inputs_from_PICMI')``
which runs the preprocessor, generating the AMReX inputs file.

* **Run** the script with Python:

.. code-block:: bash

   python <python_script>

where ``<python_script>`` is the name of the script.
This creates the Artemis input file that you can run as normal with the Artemis executable.
