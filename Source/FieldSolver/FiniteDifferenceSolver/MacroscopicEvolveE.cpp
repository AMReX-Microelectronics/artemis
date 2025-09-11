#include "FiniteDifferenceSolver.H"

#ifdef WARPX_DIM_RZ
    // currently works only for 3D
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/FieldAccessorFunctors.H"
#endif
#include "MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;

void FiniteDifferenceSolver::MacroscopicEvolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
#ifndef WARPX_MAG_LLG
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
#else
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Hfield,
#endif
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties)
{

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
#    ifndef WARPX_MAG_LLG
    amrex::ignore_unused(Efield, Bfield, Jfield, edge_lengths, dt, macroscopic_properties);
#    else
    amrex::ignore_unused(Efield, Hfield, Jfield, edge_lengths, dt, macroscopic_properties);
#endif
    WARPX_ABORT_WITH_MESSAGE("currently macro E-push does not work for RZ");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_grid_type != GridType::Collocated, "Macroscopic E field solver does not work on collocated grids");


    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {

        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {

            MacroscopicEvolveECartesian <CartesianYeeAlgorithm, LaxWendroffAlgo>
#ifndef WARPX_MAG_LLG
                       ( Efield, Bfield, Jfield, edge_lengths, dt, macroscopic_properties);
#else
                       ( Efield, Hfield, Jfield, edge_lengths, dt, macroscopic_properties);
#endif
        }
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianYeeAlgorithm, BackwardEulerAlgo>
#ifndef WARPX_MAG_LLG
                       ( Efield, Bfield, Jfield, edge_lengths, dt, macroscopic_properties);
#else
                       ( Efield, Hfield, Jfield, edge_lengths, dt, macroscopic_properties);
#endif

        }

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        // Note : EvolveE is the same for CKC and Yee.
        // In the templated Yee and CKC calls, the core operations for EvolveE is the same.
        if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, LaxWendroffAlgo>
#ifndef WARPX_MAG_LLG
                       ( Efield, Bfield, Jfield, edge_lengths, dt, macroscopic_properties);
#else
                       ( Efield, Hfield, Jfield, edge_lengths, dt, macroscopic_properties);
#endif
        } else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {

            MacroscopicEvolveECartesian <CartesianCKCAlgorithm, BackwardEulerAlgo>
#ifndef WARPX_MAG_LLG
                       ( Efield, Bfield, Jfield, edge_lengths, dt, macroscopic_properties);
#else
                       ( Efield, Hfield, Jfield, edge_lengths, dt, macroscopic_properties);
#endif
        }

    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "MacroscopicEvolveE: Unknown algorithm");
    }

#endif
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo, typename T_MacroAlgo>
void FiniteDifferenceSolver::MacroscopicEvolveECartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
#ifndef WARPX_MAG_LLG
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
#else
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Hfield,
#endif
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties)
{
#ifndef AMREX_USE_EB
    amrex::ignore_unused(edge_lengths);
#endif

    auto &warpx = WarpX::GetInstance();

    const int lev = 0;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();

    int use_lumped_resistor = warpx.use_lumped_resistor;
    int use_lumped_capacitor = warpx.use_lumped_capacitor;

    amrex::MultiFab& sigma_mf = macroscopic_properties->getsigma_mf();
    amrex::MultiFab& epsilon_mf = macroscopic_properties->getepsilon_mf();

    amrex::MultiFab* lumped_resistor_x_mf  = nullptr;
    amrex::MultiFab* lumped_resistor_y_mf  = nullptr;
    amrex::MultiFab* lumped_resistor_z_mf  = nullptr;
    amrex::MultiFab* lumped_capacitor_x_mf = nullptr;
    amrex::MultiFab* lumped_capacitor_y_mf = nullptr;
    amrex::MultiFab* lumped_capacitor_z_mf = nullptr;

    if (use_lumped_resistor){
        amrex::MultiFab& lumped_resistor_x_mf = macroscopic_properties->getlumped_resistor_x_mf();
        amrex::MultiFab& lumped_resistor_y_mf = macroscopic_properties->getlumped_resistor_y_mf();
        amrex::MultiFab& lumped_resistor_z_mf = macroscopic_properties->getlumped_resistor_z_mf();
    }

    if (use_lumped_capacitor){
        amrex::MultiFab& lumped_capacitor_x_mf = macroscopic_properties->getlumped_capacitor_x_mf();
        amrex::MultiFab& lumped_capacitor_y_mf = macroscopic_properties->getlumped_capacitor_y_mf();
        amrex::MultiFab& lumped_capacitor_z_mf = macroscopic_properties->getlumped_capacitor_z_mf();
    }

#ifndef WARPX_MAG_LLG
    amrex::MultiFab& mu_mf = macroscopic_properties->getmu_mf();
#endif

    // Index type required for calling ablastr::coarsen::sample::Interp to interpolate macroscopic
    // properties from their respective staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& sigma_stag = macroscopic_properties->sigma_IndexType;
    amrex::GpuArray<int, 3> const& epsilon_stag = macroscopic_properties->epsilon_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr     = macroscopic_properties->macro_cr_ratio;

    if (use_lumped_resistor){
        amrex::GpuArray<int, 3> const& lumped_resistor_x_stag = macroscopic_properties->lumped_resistor_x_IndexType;
        amrex::GpuArray<int, 3> const& lumped_resistor_y_stag = macroscopic_properties->lumped_resistor_y_IndexType;
        amrex::GpuArray<int, 3> const& lumped_resistor_z_stag = macroscopic_properties->lumped_resistor_z_IndexType;
    }

    if (use_lumped_capacitor){
        amrex::GpuArray<int, 3> const& lumped_capacitor_x_stag = macroscopic_properties->lumped_capacitor_x_IndexType;
        amrex::GpuArray<int, 3> const& lumped_capacitor_y_stag = macroscopic_properties->lumped_capacitor_y_IndexType;
        amrex::GpuArray<int, 3> const& lumped_capacitor_z_stag = macroscopic_properties->lumped_capacitor_z_IndexType;
    }

    amrex::GpuArray<int, 3> const& Ex_stag = macroscopic_properties->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = macroscopic_properties->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = macroscopic_properties->Ez_IndexType;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& jx = Jfield[0]->array(mfi);
        Array4<Real> const& jy = Jfield[1]->array(mfi);
        Array4<Real> const& jz = Jfield[2]->array(mfi);
#ifndef WARPX_MAG_LLG
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
#endif

#ifdef AMREX_USE_EB
        amrex::Array4<amrex::Real> const& lx = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const& ly = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#endif

        // material prop //
        amrex::Array4<amrex::Real> const& sigma_arr = sigma_mf.array(mfi);
        amrex::Array4<amrex::Real> const& eps_arr = epsilon_mf.array(mfi);

        // Lumped element arrays (default-constructed; set only if feature enabled)
        amrex::Array4<amrex::Real> resistor_x_arr, resistor_y_arr, resistor_z_arr;
        amrex::Array4<amrex::Real> capacitor_x_arr, capacitor_y_arr, capacitor_z_arr;

        if (use_lumped_resistor) {
            // These produce lightweight views; safe to copy and capture by value
            resistor_x_arr = lumped_resistor_x_mf->array(mfi);
            resistor_y_arr = lumped_resistor_y_mf->array(mfi);
            resistor_z_arr = lumped_resistor_z_mf->array(mfi);
        }
        if (use_lumped_capacitor) {
            capacitor_x_arr = lumped_capacitor_x_mf->array(mfi);
            capacitor_y_arr = lumped_capacitor_y_mf->array(mfi);
            capacitor_z_arr = lumped_capacitor_z_mf->array(mfi);
        }

        
#ifndef WARPX_MAG_LLG
        amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);
#endif

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

#ifndef WARPX_MAG_LLG
        // This functor computes Hx = Bx/mu
        // Note that mu is cell-centered here and will be interpolated/averaged
        // to the location where the B-field and H-field are defined
        FieldAccessorMacroscopic const Hx(Bx, mu_arr);
        FieldAccessorMacroscopic const Hy(By, mu_arr);
        FieldAccessorMacroscopic const Hz(Bz, mu_arr);
#else
        Array4<Real> const& Hx = Hfield[0]->array(mfi);
        Array4<Real> const& Hy = Hfield[1]->array(mfi);
        Array4<Real> const& Hz = Hfield[2]->array(mfi);
#endif

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());
        // starting component to interpolate macro properties to Ex, Ey, Ez locations
        const int scomp = 0;
        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lx(i, j, k) <= 0) return;
#endif
                // Interpolate conductivity, sigma, to Ex position on the grid
                amrex::Real const sigma_interp = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag,
                                                                                  Ex_stag, macro_cr, i, j, k, scomp);
                // Interpolated permittivity, epsilon, to Ex position on the grid
                amrex::Real const epsilon_interp = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag,
                                                                                    Ex_stag, macro_cr, i, j, k, scomp);

                // Extra conductance term from a lumped resistor on Ex edges
                amrex::Real extra_sigma = 0._rt;
                if (use_lumped_resistor) {
                    amrex::Real const R = resistor_x_arr(i,j,k);
                    if (R != 0._rt) {
                        // edge length along x over cross-section area
                        extra_sigma = dx[0] / (dx[1]*dx[2]*R);
                    }
                }
                amrex::Real fac1 = dt / epsilon_interp * (sigma_interp + extra_sigma);

                amrex::Real fac2 = 0._rt;
                if (use_lumped_capacitor) {
                    amrex::Real const C = capacitor_x_arr(i,j,k);
                    if (C != 0._rt) {
                        fac2 = C * dx[0] / (dx[1]*dx[2]*epsilon_interp);
                    }
                }
                
                amrex::Real alpha_compact = T_MacroAlgo::alpha_compact(fac1, fac2);
                amrex::Real beta_compact = T_MacroAlgo::beta_compact(fac1, fac2);

                Ex(i, j, k) = alpha_compact * Ex(i, j, k)
                            + dt/epsilon_interp * beta_compact * ( - T_Algo::DownwardDz(Hy, coefs_z, n_coefs_z, i, j, k,0)
                                       + T_Algo::DownwardDy(Hz, coefs_y, n_coefs_y, i, j, k,0)) 
                            - dt/epsilon_interp * beta_compact * jx(i, j, k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
#ifdef WARPX_DIM_3D
                if (ly(i,j,k) <= 0) return;
#elif defined(WARPX_DIM_XZ)
                //In XZ Ey is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(ly);
                if (lx(i, j, k)<=0 || lx(i-1, j, k)<=0 || lz(i, j, k)<=0 || lz(i, j-1, k)<=0) return;
#endif
#endif
                // Interpolate conductivity, sigma, to Ey position on the grid
                amrex::Real const sigma_interp = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag,
                                                                                  Ey_stag, macro_cr, i, j, k, scomp);
                // Interpolated permittivity, epsilon, to Ey position on the grid
                amrex::Real const epsilon_interp = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag,
                                                                                    Ey_stag, macro_cr, i, j, k, scomp);

                amrex::Real extra_sigma = 0._rt;
                if (use_lumped_resistor) {
                    amrex::Real const R = resistor_y_arr(i,j,k);
                    if (R != 0._rt) {
                        extra_sigma = dx[1] / (dx[0]*dx[2]*R);
                    }
                }
                amrex::Real const fac1 = (dt/epsilon_interp) * (sigma_interp + extra_sigma);

                amrex::Real fac2 = 0._rt;
                if (use_lumped_capacitor) {
                    amrex::Real const C = capacitor_y_arr(i,j,k);
                    if (C != 0._rt) {
                        fac2 = C * dx[1] / (dx[0]*dx[2]*epsilon_interp);
                    }
                }

                amrex::Real alpha_compact = T_MacroAlgo::alpha_compact(fac1, fac2);
                amrex::Real beta_compact = T_MacroAlgo::beta_compact(fac1, fac2);

                Ey(i, j, k) = alpha_compact * Ey(i, j, k)
                            + dt/epsilon_interp * beta_compact * ( - T_Algo::DownwardDx(Hz, coefs_x, n_coefs_x, i, j, k,0)
                                       + T_Algo::DownwardDz(Hx, coefs_z, n_coefs_z, i, j, k,0)) 
                            - dt/epsilon_interp * beta_compact * jy(i, j, k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lz(i,j,k) <= 0) return;
#endif
                // Interpolate conductivity, sigma, to Ez position on the grid
                amrex::Real const sigma_interp = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag,
                                                                                  Ez_stag, macro_cr, i, j, k, scomp);
                // Interpolated permittivity, epsilon, to Ez position on the grid
                amrex::Real const epsilon_interp = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag,
                                                                                    Ez_stag, macro_cr, i, j, k, scomp);

                amrex::Real extra_sigma = 0._rt;
                if (use_lumped_resistor) {
                    amrex::Real const R = resistor_z_arr(i,j,k);
                    if (R != 0._rt) {
                        extra_sigma = dx[2] / (dx[0]*dx[1]*R);
                    }
                }
                amrex::Real const fac1 = (dt/epsilon_interp) * (sigma_interp + extra_sigma);

                amrex::Real fac2 = 0._rt;
                if (use_lumped_capacitor) {
                    amrex::Real const C = capacitor_z_arr(i,j,k);
                    if (C != 0._rt) {
                        fac2 = C * dx[2] / (dx[0]*dx[1]*epsilon_interp);
                    }
                }
                
                amrex::Real alpha_compact = T_MacroAlgo::alpha_compact(fac1, fac2);
                amrex::Real beta_compact = T_MacroAlgo::beta_compact(fac1, fac2);

                Ez(i, j, k) = alpha_compact * Ez(i, j, k)
                            + dt/epsilon_interp * beta_compact * ( - T_Algo::DownwardDy(Hx, coefs_y, n_coefs_y, i, j, k,0)
                                       + T_Algo::DownwardDx(Hy, coefs_x, n_coefs_x, i, j, k,0)) 
                            - dt/epsilon_interp * beta_compact * jz(i, j, k);
            }
        );
    }
}

#endif // corresponds to ifndef WARPX_DIM_RZ
