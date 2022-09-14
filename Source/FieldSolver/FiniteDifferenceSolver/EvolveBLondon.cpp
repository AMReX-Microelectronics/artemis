/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#ifndef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;

/**
 * \brief Update the B field, over one timestep
 */
void FiniteDifferenceSolver::EvolveBLondon (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& current,
    std::unique_ptr<amrex::MultiFab> const& Gfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& face_areas,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& /* area_mod */,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& /* ECTRhofield */,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& /* Venl */,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& /* flag_info_cell */,
    std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 >& /* borrowing */,
    int lev, amrex::Real const dt, amrex::Real const penetration_depth ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Bfield, current, Gfield, face_areas,
                         lev, dt, penetration_depth);
    amrex::Abort("EvolveBLondon: RZ not implemented");
#else
    if(m_do_nodal or m_fdtd_algo != MaxwellSolverAlgo::ECT){
        amrex::ignore_unused(face_areas);
    }

    if (m_do_nodal) {

        EvolveBLondonCartesian <CartesianNodalAlgorithm> ( Bfield, current, Gfield, lev, dt, penetration_depth);

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveBLondonCartesian <CartesianYeeAlgorithm> ( Bfield, current, Gfield, lev, dt, penetration_depth );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveBLondonCartesian <CartesianCKCAlgorithm> ( Bfield, current, Gfield, lev, dt, penetration_depth );
    } else {
        amrex::Abort("EvolveBLondon: Unknown algorithm");
    }
#endif
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveBLondonCartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& current,
    std::unique_ptr<amrex::MultiFab> const& /* Gfield */,
    int lev, amrex::Real const /* dt */, amrex::Real const penetration_depth ) {

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    amrex::Real const lambdasq_mu0_fac = penetration_depth * penetration_depth * PhysConst::mu0;
    amrex::Print() << lambdasq_mu0_fac << "\n";
    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = amrex::second();

        // Extract field data for this grid/tile
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& jx = current[0]->array(mfi);
        Array4<Real> const& jy = current[1]->array(mfi);
        Array4<Real> const& jz = current[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Bx(i, j, k) =  (T_Algo::UpwardDz(jy, coefs_z, n_coefs_z, i, j, k)
                             - T_Algo::UpwardDy(jz, coefs_y, n_coefs_y, i, j, k) )
                             * lambdasq_mu0_fac;

            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                By(i, j, k) = ( T_Algo::UpwardDx(jz, coefs_x, n_coefs_x, i, j, k)
                             - T_Algo::UpwardDz(jx, coefs_z, n_coefs_z, i, j, k) )
                              * lambdasq_mu0_fac;

            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Bz(i, j, k) = ( T_Algo::UpwardDy(jx, coefs_y, n_coefs_y, i, j, k)
                             - T_Algo::UpwardDx(jy, coefs_x, n_coefs_x, i, j, k))
                              * lambdasq_mu0_fac;

            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = amrex::second() - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


#else // corresponds to ifndef WARPX_DIM_RZ

#endif // corresponds to ifndef WARPX_DIM_RZ
