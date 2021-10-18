#include "Utils/WarpXAlgorithmSelection.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#ifdef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/FieldAccessorFunctors.H"
#endif
#include "BoundaryConditions/PML.H"
#include "BoundaryConditions/PML_current.H"
#include "BoundaryConditions/PMLComponent.H"
#include "Utils/CoarsenIO.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>
#include <AMReX.H>

using namespace amrex;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::MacroscopicEvolveEPML (
    std::array< amrex::MultiFab*, 3 > Efield,
#ifndef WARPX_MAG_LLG
    std::array< amrex::MultiFab*, 3 > const Bfield,
#else
    std::array< amrex::MultiFab*, 3 > const Hfield,
#endif
    std::array< amrex::MultiFab*, 3 > const Jfield,
    amrex::MultiFab* const Ffield,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    const int lev) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
#    ifndef WARPX_MAG_LLG
    amrex::ignore_unused(Efield, Bfield, Jfield, Ffield, sigba, dt, pml_has_particles, macroscopic_properties, lev);
#    else
    amrex::ignore_unused(Efield, Hfield, Jfield, Ffield, sigba, dt, pml_has_particles, macroscopic_properties, lev);
#    endif
    amrex::Abort("PML are not implemented in cylindrical geometry.");
#else
    if (m_do_nodal) {

        amrex::Abort("Macro E-push is not implemented for nodal, yet.");

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {
            MacroscopicEvolveEPMLCartesian <CartesianYeeAlgorithm, LaxWendroffAlgo> (
                Efield,
#ifndef WARPX_MAG_LLG
                Bfield,
#else
                Hfield,
#endif
                Jfield, Ffield, sigba, dt, pml_has_particles,
                macroscopic_properties, lev);
        }
        else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {
            MacroscopicEvolveEPMLCartesian <CartesianYeeAlgorithm, BackwardEulerAlgo> (
                Efield,
#ifndef WARPX_MAG_LLG
                Bfield,
#else
                Hfield,
#endif
                Jfield, Ffield, sigba, dt, pml_has_particles,
                macroscopic_properties, lev);
        }

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {
        // Note :: Macroscopic Evolve E for PML is the same for CKC and Yee
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {
            MacroscopicEvolveEPMLCartesian <CartesianCKCAlgorithm, LaxWendroffAlgo> (
                Efield,
#ifndef WARPX_MAG_LLG
                Bfield,
#else
                Hfield,
#endif
                Jfield, Ffield, sigba, dt, pml_has_particles,
                macroscopic_properties, lev);
        }
        else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {
            MacroscopicEvolveEPMLCartesian <CartesianCKCAlgorithm, BackwardEulerAlgo> (
                Efield,
#ifndef WARPX_MAG_LLG
                Bfield,
#else
                Hfield,
#endif
                Jfield, Ffield, sigba, dt, pml_has_particles,
                macroscopic_properties, lev );
        }

    } else {
        amrex::Abort("Unknown algorithm");
    }
#endif
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo, typename T_MacroAlgo>
void FiniteDifferenceSolver::MacroscopicEvolveEPMLCartesian (
    std::array< amrex::MultiFab*, 3 > Efield,
#ifndef WARPX_MAG_LLG
    std::array< amrex::MultiFab*, 3 > const Bfield,
#else
    std::array< amrex::MultiFab*, 3 > const Hfield,
#endif
    std::array< amrex::MultiFab*, 3 > const Jfield,
    amrex::MultiFab* const Ffield,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    const int lev ) {

    amrex::ignore_unused(Ffield);
    // Index type required for calling CoarsenIO::Interp to interpolate macroscopic
    // properties from their respective staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Ex_stag = macroscopic_properties->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = macroscopic_properties->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = macroscopic_properties->Ez_IndexType;

    const auto getSigma = GetSigmaMacroparameter();
    const auto getEpsilon = GetEpsilonMacroparameter();
    const auto getMu = GetMuMacroparameter();
    auto &warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const auto dx = warpx.Geom(lev).CellSizeArray();


    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
#ifndef WARPX_MAG_LLG
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
#endif


        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

#ifndef WARPX_MAG_LLG
        FieldAccessorMacroscopic<GetMuMacroparameter> const Hx(Bx, getMu, Bx_stag, problo, dx);
        FieldAccessorMacroscopic<GetMuMacroparameter> const Hy(By, getMu, By_stag, problo, dx);
        FieldAccessorMacroscopic<GetMuMacroparameter> const Hz(Bz, getMu, Bz_stag, problo, dx);
#else
        Array4<Real> const& Hx = Hfield[0]->array(mfi);
        Array4<Real> const& Hy = Hfield[1]->array(mfi);
        Array4<Real> const& Hz = Hfield[2]->array(mfi);
#endif

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates (i, j, k, Ex_stag, problo, dx,
                                                   x, y, z );
                amrex::Real const sigma = getSigma(x, y, z);
                amrex::Real const epsilon = getEpsilon(x, y, z);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma, epsilon, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);
                Ex(i, j, k, PMLComp::xz) = alpha * Ex(i, j, k, PMLComp::xz) - beta * (
                    T_Algo::DownwardDz(Hy, coefs_z, n_coefs_z, i, j, k, PMLComp::yx)
                  + T_Algo::DownwardDz(Hy, coefs_z, n_coefs_z, i, j, k, PMLComp::yz) );
                Ex(i, j, k, PMLComp::xy) = alpha * Ex(i, j, k, PMLComp::xy) + beta * (
                    T_Algo::DownwardDy(Hz, coefs_y, n_coefs_y, i, j, k, PMLComp::zx)
                  + T_Algo::DownwardDy(Hz, coefs_y, n_coefs_y, i, j, k, PMLComp::zy) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates (i, j, k, Ey_stag, problo, dx,
                                                   x, y, z );
                amrex::Real const sigma = getSigma(x, y, z);
                amrex::Real const epsilon = getEpsilon(x, y, z);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma, epsilon, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);

                Ey(i, j, k, PMLComp::yx) = alpha * Ey(i, j, k, PMLComp::yx) - beta * (
                    T_Algo::DownwardDx(Hz, coefs_x, n_coefs_x, i, j, k, PMLComp::zx)
                  + T_Algo::DownwardDx(Hz, coefs_x, n_coefs_x, i, j, k, PMLComp::zy) );
                Ey(i, j, k, PMLComp::yz) = alpha * Ey(i, j, k, PMLComp::yz) + beta * (
                    T_Algo::DownwardDz(Hx, coefs_z, n_coefs_z, i, j, k, PMLComp::xy)
                  + T_Algo::DownwardDz(Hx, coefs_z, n_coefs_z, i, j, k, PMLComp::xz) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates (i, j, k, Ez_stag, problo, dx,
                                                   x, y, z );
                amrex::Real const sigma = getSigma(x, y, z);
                amrex::Real const epsilon = getEpsilon(x, y, z);
                amrex::Real alpha = T_MacroAlgo::alpha( sigma, epsilon, dt);
                amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);

                Ez(i, j, k, PMLComp::zy) = alpha * Ez(i, j, k, PMLComp::zy) - beta * (
                    T_Algo::DownwardDy(Hx, coefs_y, n_coefs_y, i, j, k, PMLComp::xy)
                  + T_Algo::DownwardDy(Hx, coefs_y, n_coefs_y, i, j, k, PMLComp::xz) );
                Ez(i, j, k, PMLComp::zx) = alpha * Ez(i, j, k, PMLComp::zx) + beta * (
                    T_Algo::DownwardDx(Hy, coefs_x, n_coefs_x, i, j, k, PMLComp::yx)
                  + T_Algo::DownwardDx(Hy, coefs_x, n_coefs_x, i, j, k, PMLComp::yz) );
            }

        );

        // Update the E field in the PML, using the current
        // deposited by the particles in the PML
        if (pml_has_particles) {

            // Extract field data for this grid/tile
            Array4<Real> const& Jx = Jfield[0]->array(mfi);
            Array4<Real> const& Jy = Jfield[1]->array(mfi);
            Array4<Real> const& Jz = Jfield[2]->array(mfi);
            const Real* sigmaj_x = sigba[mfi].sigma[0].data();
            const Real* sigmaj_y = sigba[mfi].sigma[1].data();
            const Real* sigmaj_z = sigba[mfi].sigma[2].data();
            int const x_lo = sigba[mfi].sigma[0].lo();
#if (AMREX_SPACEDIM == 3)
            int const y_lo = sigba[mfi].sigma[1].lo();
            int const z_lo = sigba[mfi].sigma[2].lo();
#else
            int const y_lo = 0;
            int const z_lo = sigba[mfi].sigma[1].lo();
#endif

            amrex::ParallelFor( tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    amrex::Real x, y, z;
                    WarpXUtilAlgo::getCellCoordinates (i, j, k, Ex_stag, problo, dx,
                                                       x, y, z );
                    amrex::Real const sigma = getSigma(x, y, z);
                    amrex::Real const epsilon = getEpsilon(x, y, z);
                    amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);
                    push_ex_pml_current(i, j, k, Ex, Jx,
                        sigmaj_y, sigmaj_z, y_lo, z_lo, beta);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    amrex::Real x, y, z;
                    WarpXUtilAlgo::getCellCoordinates (i, j, k, Ey_stag, problo, dx,
                                                       x, y, z );
                    amrex::Real const sigma = getSigma(x, y, z);
                    amrex::Real const epsilon = getEpsilon(x, y, z);
                    amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);
                    push_ey_pml_current(i, j, k, Ey, Jy,
                        sigmaj_x, sigmaj_z, x_lo, z_lo, beta);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    amrex::Real x, y, z;
                    WarpXUtilAlgo::getCellCoordinates (i, j, k, Ez_stag, problo, dx,
                                                       x, y, z );
                    amrex::Real const sigma = getSigma(x, y, z);
                    amrex::Real const epsilon = getEpsilon(x, y, z);
                    amrex::Real beta = T_MacroAlgo::beta( sigma, epsilon, dt);
                    push_ez_pml_current(i, j, k, Ez, Jz,
                        sigmaj_x, sigmaj_y, x_lo, y_lo, beta);
                }
            );
        }

    }

}

#endif // corresponds to ifndef WARPX_DIM_RZ
