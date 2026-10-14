/* Copyright 2022 David Grote
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_ACCELERATORLATTICE_LATTICEELEMENTS_DRIFT_H_
#define ARTEMIS_ACCELERATORLATTICE_LATTICEELEMENTS_DRIFT_H_

#include "LatticeElementBase.H"

// Specifies a drift

struct Drift
    : LatticeElementBase
{

    Drift ();

    ~Drift () = default;

    /**
     * \brief Read in an element and add it to the lists
     *
     * @param[in] pp_element The ParmParse instance to read in the data
     * @param[inout] z_location The current z location in the lattice
     */
    void
    AddElement (amrex::ParmParse & pp_element, amrex::ParticleReal & z_location);

};

#endif // ARTEMIS_ACCELERATORLATTICE_LATTICEELEMENTS_DRIFT_H_
