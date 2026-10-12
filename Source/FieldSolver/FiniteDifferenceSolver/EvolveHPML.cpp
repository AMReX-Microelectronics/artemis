/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "BoundaryConditions/PMLComponent.H"
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX.H>

using namespace amrex;
#ifndef WARPX_DIM_RZ
#ifdef WARPX_MAG_LLG
/**
 * \brief Update the H field, over one timestep
 */
void FiniteDifferenceSolver::EvolveHPML (
    std::array< amrex::MultiFab*, 3 > Hfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    amrex::Real const dt,
    const bool dive_cleaning) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Hfield, Efield, dt, dive_cleaning);
    amrex::Abort("PML are not implemented in cylindrical geometry.");
#else
    if (m_grid_type == GridType::Collocated) {

        EvolveHPMLCartesian <CartesianNodalAlgorithm> (Hfield, Efield, dt, dive_cleaning);

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {

        EvolveHPMLCartesian <CartesianYeeAlgorithm> (Hfield, Efield, dt, dive_cleaning);

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveHPMLCartesian <CartesianCKCAlgorithm> (Hfield, Efield, dt, dive_cleaning);

    } else {
        amrex::Abort("EvolveHPML: Unknown algorithm");
    }
#endif
}


template<typename T_Algo>
void FiniteDifferenceSolver::EvolveHPMLCartesian (
    std::array< amrex::MultiFab*, 3 > Hfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    amrex::Real const dt,
    const bool dive_cleaning) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Hx = Hfield[0]->array(mfi);
        Array4<Real> const& Hy = Hfield[1]->array(mfi);
        Array4<Real> const& Hz = Hfield[2]->array(mfi);
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Hfield[0]->ixType().ixType());
        Box const& tby  = mfi.tilebox(Hfield[1]->ixType().ixType());
        Box const& tbz  = mfi.tilebox(Hfield[2]->ixType().ixType());

        amrex::Real mu0_inv = 1._rt/PhysConst::mu0;

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                amrex::Real UpwardDz_Ey_yy = 0._rt;
                amrex::Real UpwardDy_Ez_zz = 0._rt;
                if (dive_cleaning)
                {
                    UpwardDz_Ey_yy = T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yy);
                    UpwardDy_Ez_zz = T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zz);
                }

                Hx(i, j, k, PMLComp::xz) += mu0_inv * dt * (
                    T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yx)
                  + T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yz)
                  + UpwardDz_Ey_yy);

                Hx(i, j, k, PMLComp::xy) -= mu0_inv * dt * (
                    T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zx)
                  + T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zy)
                  + UpwardDy_Ez_zz);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                amrex::Real UpwardDx_Ez_zz = 0._rt;
                amrex::Real UpwardDz_Ex_xx = 0._rt;
                if (dive_cleaning)
                {
                    UpwardDx_Ez_zz = T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zz);
                    UpwardDz_Ex_xx = T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xx);
                }

                Hy(i, j, k, PMLComp::yx) += mu0_inv * dt * (
                    T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zx)
                  + T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zy)
                  + UpwardDx_Ez_zz);

                Hy(i, j, k, PMLComp::yz) -= mu0_inv * dt * (
                    UpwardDz_Ex_xx
                  + T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xy)
                  + T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xz));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                amrex::Real UpwardDy_Ex_xx = 0._rt;
                amrex::Real UpwardDx_Ey_yy = 0._rt;
                if (dive_cleaning)
                {
                    UpwardDy_Ex_xx = T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xx);
                    UpwardDx_Ey_yy = T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yy);
                }

                Hz(i, j, k, PMLComp::zy) += mu0_inv * dt * (
                    UpwardDy_Ex_xx
                  + T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xy)
                  + T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xz) );

                Hz(i, j, k, PMLComp::zx) -= mu0_inv * dt * (
                    T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yx)
                  + T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yz)
                  + UpwardDx_Ey_yy);
            }

        );

    }

}

#endif // corresponds to ifdef WARPX_MAG_LLG
#endif // corresponds to ifndef WARPX_DIM_RZ
