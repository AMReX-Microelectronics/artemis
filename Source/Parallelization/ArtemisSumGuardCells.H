/* Copyright 2019 Maxence Thevenet, Remi Lehe, Weiqun Zhang
 *
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_SUM_GUARD_CELLS_H_
#define ARTEMIS_SUM_GUARD_CELLS_H_

#include "Utils/ArtemisAlgorithmSelection.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_MultiFab.H>

/** \brief Sum the values of `mf`, where the different boxes overlap
 * (i.e. in the guard cells)
 *
 * This is typically called for the sources of the Maxwell equations (J/rho)
 * after deposition from the macroparticles.
 *
 *  - When Artemis is used with a finite-difference scheme: this only
 *    updates the *valid* cells of `mf`
 *  - When Artemis is used with a spectral scheme (PSATD): this
 *    updates both the *valid* cells and *guard* cells. (This is because a
 *    spectral solver requires the value of the sources over a large stencil.)
 */
inline void
ArtemisSumGuardCells(amrex::MultiFab& mf, const amrex::Periodicity& period,
                   const amrex::IntVect& src_ngrow,
                   const int icomp=0, const int ncomp=1)
{
    amrex::IntVect n_updated_guards;

    // Update both valid cells and guard cells
    if (Artemis::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        n_updated_guards = mf.nGrowVect();
    else  // Update only the valid cells
        n_updated_guards = amrex::IntVect::TheZeroVector();
    ablastr::utils::communication::SumBoundary(mf, icomp, ncomp, src_ngrow, n_updated_guards, Artemis::do_single_precision_comms, period);
}

/** \brief Sum the values of `src` where the different boxes overlap
 * (i.e. in the guard cells) and copy them into `dst`
 *
 * This is typically called for the sources of the Maxwell equations (J/rho)
 * after deposition from the macroparticles + filtering.
 *
 *  - When Artemis is used with a finite-difference scheme: this only
 *    updates the *valid* cells of `dst`
 *  - When Artemis is used with a spectral scheme (PSATD): this
 *    updates both the *valid* cells and *guard* cells. (This is because a
 *    spectral solver requires the value of the sources over a large stencil.)
 *
 * Note: `i_comp` is the component where the results will be stored in `dst`;
 *       The component from which we copy in `src` is always 0.
 */
inline void
ArtemisSumGuardCells(amrex::MultiFab& dst, amrex::MultiFab& src,
                   const amrex::Periodicity& period,
                   const amrex::IntVect& src_ngrow,
                   const int icomp=0, const int ncomp=1)
{
    amrex::IntVect n_updated_guards;

    // Update both valid cells and guard cells
    if (Artemis::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        n_updated_guards = dst.nGrowVect();
    else  // Update only the valid cells
        n_updated_guards = amrex::IntVect::TheZeroVector();

    dst.setVal(0., icomp, ncomp, n_updated_guards);
//    ablastr::utils::communication::ParallelAdd(dst, src, 0, icomp, ncomp, src_ngrow, n_updated_guards, period);
    dst.ParallelAdd(src, 0, icomp, ncomp, src_ngrow, n_updated_guards, period);
}

#endif // ARTEMIS_SUM_GUARD_CELLS_H_
