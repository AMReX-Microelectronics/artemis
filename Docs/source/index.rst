:orphan:

ARTEMIS
-------

ARTEMIS is an advanced electrodynamics code based on `WarpX <https://ecp-warpx.github.io>`__.
It couples Maxwell's equations with classical models describing quantum behavior of materials used in microelectronics.

It supports many features including:

    - Perfectly-Matched Layers (PML)
    - Heterogeneous materials
    - User-defined excitations
    - Landau-Lifshitz-Gilbert equations for micromagenetics

For details on the algorithms that ARTEMIS implements, see the :ref:`theory section <theory>`.

ARTEMIS is a *highly-parallel and highly-optimized code*, which can run on GPUs and multi-core CPUs, and includes load balancing capabilities.
In addition, ARTEMIS is also a *multi-platform code* and runs on Linux, macOS and Windows. ARTEMIS has leveraged the ECP code WarpX, and is built on the ECP framework AMReX.

.. _contact:

Contact us
^^^^^^^^^^

The `ARTEMIS GitHub repo <https://github.com/ECP-WarpX/artemis>`__ is the main communication platform.
Have a look at the action icons on the top right of the web page: feel free to watch the repo if you want to receive updates, or to star the repo to support the project.
For bug reports or to request new features, you can also open a new `issue <https://github.com/ECP-WarpX/artemis/issues>`__.

We also have a `discussion page <https://github.com/ECP-WarpX/artemis/discussions>`__ on which you can find already answered questions, add new questions, get help with installation procedures, discuss ideas or share comments.

.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#installation,
   section#usage,
   section#theory,
   section#data-analysis,
   section#development,
   section#maintenance,
   section#epilogue {
       display:none;
   }
   </style>

.. toctree::
   :hidden:

   coc
   acknowledge_us
   highlights

Installation
------------
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/users
   install/cmake
   install/hpc
..   install/changelog
..   install/upgrade

Usage
-----
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/how_to_run
   usage/domain_decomposition
   usage/parameters
   usage/python
   usage/examples
   usage/pwfa
   usage/workflows
   usage/faq

Data Analysis
-------------
.. toctree::
   :caption: DATA ANALYSIS
   :maxdepth: 1
   :hidden:

   dataanalysis/formats
   dataanalysis/yt
   dataanalysis/openpmdviewer
   dataanalysis/openpmdapi
   dataanalysis/paraview
   dataanalysis/visit
   dataanalysis/visualpic
   dataanalysis/picviewer
   dataanalysis/reduced_diags
   dataanalysis/workflows

Theory
------
.. toctree::
   :caption: THEORY
   :maxdepth: 1
   :hidden:

   theory/intro
   theory/picsar_theory
   theory/amr
   theory/PML
   theory/boosted_frame
   theory/input_output
   theory/collisions

Development
-----------
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   developers/contributing
   developers/workflows
   developers/developers
   developers/doxygen
   developers/gnumake
   developers/faq
.. good to have in the future:
..   developers/repostructure

Maintenance
-----------
.. toctree::
   :caption: MAINTENANCE
   :maxdepth: 1
   :hidden:

   maintenance/release
   maintenance/performance_tests

Epilogue
--------
.. toctree::
   :caption: EPILOGUE
   :maxdepth: 1
   :hidden:

   glossary
   acknowledgements
