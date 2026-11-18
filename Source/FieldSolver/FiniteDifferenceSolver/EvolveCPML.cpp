/* Convolutional PML (CPML) field updates for the Cartesian Yee FDTD solver.
 *
 * These are used when warpx.pml_type = convolutional. The PML field MultiFabs keep
 * the same layout as for the split-field PML (2 or 3 components), so that the
 * existing PML <-> domain exchange, FillBoundary, checkpointing, PEC handling, etc.
 * can be reused unchanged: with the CPML the *total* field is stored in component 0
 * and component 1 is identically zero (PML::Exchange sums the components, and sets
 * component 1 to zero in the guard cells it fills from the regular grid).
 *
 * See BoundaryConditions/CPML_K.H for the formulation.
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"

#include "BoundaryConditions/CPML_K.H"
#include "BoundaryConditions/PML.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#ifndef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/FieldAccessorFunctors.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <array>

using namespace amrex;

namespace {

#ifndef WARPX_DIM_RZ
    void check_cpml_algo (int fdtd_algo, short grid_type)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            fdtd_algo == ElectromagneticSolverAlgo::Yee && grid_type != GridType::Collocated,
            "Convolutional PML is only implemented for the staggered Yee solver");
    }

    /** B (or H) update in the CPML: F -= fac * dt * curl_s(E).
     *  fac = 1 for B, fac = 1/mu0 for H (LLG).
     *
     *  B is advanced in two half steps (FirstHalf with E^n, SecondHalf with E^{n+1}),
     *  which together form the leapfrog step B^{n-1/2} -> B^{n+1/2} driven by E^n.
     *  The memory variable is therefore updated once per leapfrog step, at the start
     *  of the SecondHalf (psi^{n+1} = b psi^n + a dE^{n+1}, with the coefficients of
     *  the full step 2*dt), and that same psi is used again by the following FirstHalf
     *  (E, hence dE, is unchanged in between). This makes the CPML recursion
     *  consistent with leapfrog; updating psi at every half step with half-step
     *  coefficients is noticeably more reflective.
     *  A DtType::Full push updates psi with dt and applies it over dt. */
    void EvolveBorHCPMLYee (
        std::array< amrex::MultiFab*, 3 > Bfield,
        std::array< amrex::MultiFab*, 3 > const Efield,
        std::array< amrex::MultiFab*, 3 > const psiB,
        MultiSigmaBox const& sigba,
        amrex::Real const a_dt, DtType const dt_type, amrex::Real const fac,
        amrex::Real const * const coefs_x_dev, int const n_coefs_x,
        amrex::Real const * const coefs_y_dev, int const n_coefs_y,
        amrex::Real const * const coefs_z_dev, int const n_coefs_z)
    {
        using Yee = CartesianYeeAlgorithm;

        // time step applied to B, and time step of the psi recursion
        amrex::Real const dt = a_dt;
        bool const update_psi = (dt_type != DtType::FirstHalf);
        amrex::Real const dt_psi = (dt_type == DtType::SecondHalf) ? 2._rt*a_dt : a_dt;

        auto const Bx_stag = cpml_stag(Bfield[0]->ixType());
        auto const By_stag = cpml_stag(Bfield[1]->ixType());
        auto const Bz_stag = cpml_stag(Bfield[2]->ixType());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            Array4<Real> const& Bx = Bfield[0]->array(mfi);
            Array4<Real> const& By = Bfield[1]->array(mfi);
            Array4<Real> const& Bz = Bfield[2]->array(mfi);
            Array4<Real> const& Ex = Efield[0]->array(mfi);
            Array4<Real> const& Ey = Efield[1]->array(mfi);
            Array4<Real> const& Ez = Efield[2]->array(mfi);
            Array4<Real> const& Px = psiB[0]->array(mfi);
            Array4<Real> const& Py = psiB[1]->array(mfi);
            Array4<Real> const& Pz = psiB[2]->array(mfi);

            CPMLCoefs const cc(sigba[mfi]);

            Real const * const AMREX_RESTRICT coefs_x = coefs_x_dev;
            Real const * const AMREX_RESTRICT coefs_y = coefs_y_dev;
            Real const * const AMREX_RESTRICT coefs_z = coefs_z_dev;

            Box const& tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect());
            Box const& tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

            amrex::ParallelFor(tbx, tby, tbz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    // curl_x E = dEz/dy - dEy/dz ; psi comps: xy=0 (d/dy), xz=1 (d/dz)
                    Real const dEz_dy = Yee::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, 0);
                    Real const dEy_dz = Yee::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, 0);
                    Real const sy = cpml_stretched_derivative(cc, 1, Bx_stag[1], i, j, k, dt_psi, update_psi, dEz_dy, Px(i,j,k,0));
                    Real const sz = cpml_stretched_derivative(cc, 2, Bx_stag[2], i, j, k, dt_psi, update_psi, dEy_dz, Px(i,j,k,1));
                    Bx(i, j, k, 0) -= fac * dt * (sy - sz);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    // curl_y E = dEx/dz - dEz/dx ; psi comps: yz=0 (d/dz), yx=1 (d/dx)
                    Real const dEx_dz = Yee::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, 0);
                    Real const dEz_dx = Yee::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, 0);
                    Real const sz = cpml_stretched_derivative(cc, 2, By_stag[2], i, j, k, dt_psi, update_psi, dEx_dz, Py(i,j,k,0));
                    Real const sx = cpml_stretched_derivative(cc, 0, By_stag[0], i, j, k, dt_psi, update_psi, dEz_dx, Py(i,j,k,1));
                    By(i, j, k, 0) -= fac * dt * (sz - sx);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    // curl_z E = dEy/dx - dEx/dy ; psi comps: zx=0 (d/dx), zy=1 (d/dy)
                    Real const dEy_dx = Yee::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, 0);
                    Real const dEx_dy = Yee::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, 0);
                    Real const sx = cpml_stretched_derivative(cc, 0, Bz_stag[0], i, j, k, dt_psi, update_psi, dEy_dx, Pz(i,j,k,0));
                    Real const sy = cpml_stretched_derivative(cc, 1, Bz_stag[1], i, j, k, dt_psi, update_psi, dEx_dy, Pz(i,j,k,1));
                    Bz(i, j, k, 0) -= fac * dt * (sx - sy);
                }
            );
        }
    }
#endif // WARPX_DIM_RZ
}

void FiniteDifferenceSolver::EvolveBCPML (
    std::array< amrex::MultiFab*, 3 > Bfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    std::array< amrex::MultiFab*, 3 > const psiB,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, DtType const dt_type)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Bfield, Efield, psiB, sigba, dt, dt_type);
    WARPX_ABORT_WITH_MESSAGE("PML are not implemented in cylindrical geometry.");
#else
    check_cpml_algo(m_fdtd_algo, m_grid_type);
    EvolveBorHCPMLYee(Bfield, Efield, psiB, sigba, dt, dt_type, 1._rt,
                      m_stencil_coefs_x.dataPtr(), static_cast<int>(m_stencil_coefs_x.size()),
                      m_stencil_coefs_y.dataPtr(), static_cast<int>(m_stencil_coefs_y.size()),
                      m_stencil_coefs_z.dataPtr(), static_cast<int>(m_stencil_coefs_z.size()));
#endif
}

#ifdef WARPX_MAG_LLG
void FiniteDifferenceSolver::EvolveHCPML (
    std::array< amrex::MultiFab*, 3 > Hfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    std::array< amrex::MultiFab*, 3 > const psiH,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, DtType const dt_type)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Hfield, Efield, psiH, sigba, dt, dt_type);
    WARPX_ABORT_WITH_MESSAGE("PML are not implemented in cylindrical geometry.");
#else
    check_cpml_algo(m_fdtd_algo, m_grid_type);
    // As in EvolveHPML, the PML region is treated as vacuum for the H update (mu = mu0)
    EvolveBorHCPMLYee(Hfield, Efield, psiH, sigba, dt, dt_type, 1._rt/PhysConst::mu0,
                      m_stencil_coefs_x.dataPtr(), static_cast<int>(m_stencil_coefs_x.size()),
                      m_stencil_coefs_y.dataPtr(), static_cast<int>(m_stencil_coefs_y.size()),
                      m_stencil_coefs_z.dataPtr(), static_cast<int>(m_stencil_coefs_z.size()));
#endif
}
#endif

void FiniteDifferenceSolver::EvolveECPML (
    std::array< amrex::MultiFab*, 3 > Efield,
    std::array< amrex::MultiFab*, 3 > const Bfield,
    std::array< amrex::MultiFab*, 3 > const Jfield,
    std::array< amrex::MultiFab*, 3 > const psiE,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Efield, Bfield, Jfield, psiE, sigba, dt, pml_has_particles);
    WARPX_ABORT_WITH_MESSAGE("PML are not implemented in cylindrical geometry.");
#else
    check_cpml_algo(m_fdtd_algo, m_grid_type);
    using Yee = CartesianYeeAlgorithm;

    // dE/dt = c^2 curl_s(B) - mu0 c^2 J  (with LLG, Bfield holds H: dE/dt = curl_s(H)/eps0 - J/eps0)
    Real c2 = PhysConst::c * PhysConst::c;
#ifdef WARPX_MAG_LLG
    c2 *= PhysConst::mu0;
#endif
    Real const jfac = PhysConst::c * PhysConst::c * PhysConst::mu0; // 1/eps0

    auto const Ex_stag = cpml_stag(Efield[0]->ixType());
    auto const Ey_stag = cpml_stag(Efield[1]->ixType());
    auto const Ez_stag = cpml_stag(Efield[2]->ixType());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& Px = psiE[0]->array(mfi);
        Array4<Real> const& Py = psiE[1]->array(mfi);
        Array4<Real> const& Pz = psiE[2]->array(mfi);

        CPMLCoefs const cc(sigba[mfi]);

        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const dBz_dy = Yee::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k, 0);
                Real const dBy_dz = Yee::DownwardDz(By, coefs_z, n_coefs_z, i, j, k, 0);
                Real const sy = cpml_stretched_derivative(cc, 1, Ex_stag[1], i, j, k, dt, true, dBz_dy, Px(i,j,k,0));
                Real const sz = cpml_stretched_derivative(cc, 2, Ex_stag[2], i, j, k, dt, true, dBy_dz, Px(i,j,k,1));
                Ex(i, j, k, 0) += c2 * dt * (sy - sz);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const dBx_dz = Yee::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, 0);
                Real const dBz_dx = Yee::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, 0);
                Real const sz = cpml_stretched_derivative(cc, 2, Ey_stag[2], i, j, k, dt, true, dBx_dz, Py(i,j,k,0));
                Real const sx = cpml_stretched_derivative(cc, 0, Ey_stag[0], i, j, k, dt, true, dBz_dx, Py(i,j,k,1));
                Ey(i, j, k, 0) += c2 * dt * (sz - sx);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const dBy_dx = Yee::DownwardDx(By, coefs_x, n_coefs_x, i, j, k, 0);
                Real const dBx_dy = Yee::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k, 0);
                Real const sx = cpml_stretched_derivative(cc, 0, Ez_stag[0], i, j, k, dt, true, dBy_dx, Pz(i,j,k,0));
                Real const sy = cpml_stretched_derivative(cc, 1, Ez_stag[1], i, j, k, dt, true, dBx_dy, Pz(i,j,k,1));
                Ez(i, j, k, 0) += c2 * dt * (sx - sy);
            }
        );

        if (pml_has_particles) {
            Array4<Real> const& Jx = Jfield[0]->array(mfi);
            Array4<Real> const& Jy = Jfield[1]->array(mfi);
            Array4<Real> const& Jz = Jfield[2]->array(mfi);
            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){ Ex(i,j,k,0) -= jfac * dt * Jx(i,j,k); },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){ Ey(i,j,k,0) -= jfac * dt * Jy(i,j,k); },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){ Ez(i,j,k,0) -= jfac * dt * Jz(i,j,k); }
            );
        }
    }
#endif
}

void FiniteDifferenceSolver::MacroscopicEvolveECPML (
    std::array< amrex::MultiFab*, 3 > Efield,
    std::array< amrex::MultiFab*, 3 > const BorHfield,
    std::array< amrex::MultiFab*, 3 > const Jfield,
    std::array< amrex::MultiFab*, 3 > const psiE,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    amrex::MultiFab* const eps_mf,
    amrex::MultiFab* const mu_mf,
    amrex::MultiFab* const sigma_mf)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Efield, BorHfield, Jfield, psiE, sigba, dt, pml_has_particles,
                         macroscopic_properties, eps_mf, mu_mf, sigma_mf);
    WARPX_ABORT_WITH_MESSAGE("PML are not implemented in cylindrical geometry.");
#else
    check_cpml_algo(m_fdtd_algo, m_grid_type);
    if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::LaxWendroff) {
        MacroscopicEvolveECPMLCartesian<LaxWendroffAlgo>(
            Efield, BorHfield, Jfield, psiE, sigba, dt, pml_has_particles,
            macroscopic_properties, eps_mf, mu_mf, sigma_mf);
    } else if (WarpX::macroscopic_solver_algo == MacroscopicSolverAlgo::BackwardEuler) {
        MacroscopicEvolveECPMLCartesian<BackwardEulerAlgo>(
            Efield, BorHfield, Jfield, psiE, sigba, dt, pml_has_particles,
            macroscopic_properties, eps_mf, mu_mf, sigma_mf);
    } else {
        WARPX_ABORT_WITH_MESSAGE("MacroscopicEvolveECPML: unknown macroscopic algorithm");
    }
#endif
}

#ifndef WARPX_DIM_RZ
template<typename T_MacroAlgo>
void FiniteDifferenceSolver::MacroscopicEvolveECPMLCartesian (
    std::array< amrex::MultiFab*, 3 > Efield,
    std::array< amrex::MultiFab*, 3 > const BorHfield,
    std::array< amrex::MultiFab*, 3 > const Jfield,
    std::array< amrex::MultiFab*, 3 > const psiE,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    amrex::MultiFab* const eps_mf,
    amrex::MultiFab* const mu_mf,
    amrex::MultiFab* const sigma_mf)
{
    using Yee = CartesianYeeAlgorithm;
#ifdef WARPX_MAG_LLG
    amrex::ignore_unused(mu_mf);
#endif

    // Index types used to interpolate the (cell-centered) material properties to
    // the Ex, Ey, Ez locations, exactly as in the interior and split-field PML updates
    amrex::GpuArray<int, 3> const& sigma_stag   = macroscopic_properties->sigma_IndexType;
    amrex::GpuArray<int, 3> const& epsilon_stag = macroscopic_properties->epsilon_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr     = macroscopic_properties->macro_cr_ratio;
    amrex::GpuArray<int, 3> const& Ex_mstag = macroscopic_properties->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_mstag = macroscopic_properties->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_mstag = macroscopic_properties->Ez_IndexType;

    auto const Ex_stag = cpml_stag(Efield[0]->ixType());
    auto const Ey_stag = cpml_stag(Efield[1]->ixType());
    auto const Ez_stag = cpml_stag(Efield[2]->ixType());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Px = psiE[0]->array(mfi);
        Array4<Real> const& Py = psiE[1]->array(mfi);
        Array4<Real> const& Pz = psiE[2]->array(mfi);

        amrex::Array4<amrex::Real> const& sigma_arr = sigma_mf->array(mfi);
        amrex::Array4<amrex::Real> const& eps_arr = eps_mf->array(mfi);

#ifndef WARPX_MAG_LLG
        // H = B/mu, with mu the local permeability in the PML
        amrex::Array4<amrex::Real> const& mu_arr = mu_mf->array(mfi);
        FieldAccessorMacroscopic const Hx(BorHfield[0]->const_array(mfi), mu_arr);
        FieldAccessorMacroscopic const Hy(BorHfield[1]->const_array(mfi), mu_arr);
        FieldAccessorMacroscopic const Hz(BorHfield[2]->const_array(mfi), mu_arr);
#else
        Array4<Real> const& Hx = BorHfield[0]->array(mfi);
        Array4<Real> const& Hy = BorHfield[1]->array(mfi);
        Array4<Real> const& Hz = BorHfield[2]->array(mfi);
#endif

        CPMLCoefs const cc(sigba[mfi]);

        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

        const int scomp = 0;
        const bool has_j = pml_has_particles;
        Array4<Real const> Jx, Jy, Jz;
        if (has_j) {
            Jx = Jfield[0]->const_array(mfi);
            Jy = Jfield[1]->const_array(mfi);
            Jz = Jfield[2]->const_array(mfi);
        }

        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const sig = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag, Ex_mstag, macro_cr, i, j, k, scomp);
                Real const eps = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag, Ex_mstag, macro_cr, i, j, k, scomp);
                Real const alpha = T_MacroAlgo::alpha(sig, eps, dt);
                Real const beta  = T_MacroAlgo::beta (sig, eps, dt);

                Real const dHz_dy = Yee::DownwardDy(Hz, coefs_y, n_coefs_y, i, j, k, 0);
                Real const dHy_dz = Yee::DownwardDz(Hy, coefs_z, n_coefs_z, i, j, k, 0);
                Real const sy = cpml_stretched_derivative(cc, 1, Ex_stag[1], i, j, k, dt, true, dHz_dy, Px(i,j,k,0));
                Real const sz = cpml_stretched_derivative(cc, 2, Ex_stag[2], i, j, k, dt, true, dHy_dz, Px(i,j,k,1));
                Real const jx = has_j ? Jx(i,j,k) : 0._rt;
                Ex(i, j, k, 0) = alpha * Ex(i, j, k, 0) + beta * (sy - sz - jx);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const sig = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag, Ey_mstag, macro_cr, i, j, k, scomp);
                Real const eps = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag, Ey_mstag, macro_cr, i, j, k, scomp);
                Real const alpha = T_MacroAlgo::alpha(sig, eps, dt);
                Real const beta  = T_MacroAlgo::beta (sig, eps, dt);

                Real const dHx_dz = Yee::DownwardDz(Hx, coefs_z, n_coefs_z, i, j, k, 0);
                Real const dHz_dx = Yee::DownwardDx(Hz, coefs_x, n_coefs_x, i, j, k, 0);
                Real const sz = cpml_stretched_derivative(cc, 2, Ey_stag[2], i, j, k, dt, true, dHx_dz, Py(i,j,k,0));
                Real const sx = cpml_stretched_derivative(cc, 0, Ey_stag[0], i, j, k, dt, true, dHz_dx, Py(i,j,k,1));
                Real const jy = has_j ? Jy(i,j,k) : 0._rt;
                Ey(i, j, k, 0) = alpha * Ey(i, j, k, 0) + beta * (sz - sx - jy);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const sig = ablastr::coarsen::sample::Interp(sigma_arr, sigma_stag, Ez_mstag, macro_cr, i, j, k, scomp);
                Real const eps = ablastr::coarsen::sample::Interp(eps_arr, epsilon_stag, Ez_mstag, macro_cr, i, j, k, scomp);
                Real const alpha = T_MacroAlgo::alpha(sig, eps, dt);
                Real const beta  = T_MacroAlgo::beta (sig, eps, dt);

                Real const dHy_dx = Yee::DownwardDx(Hy, coefs_x, n_coefs_x, i, j, k, 0);
                Real const dHx_dy = Yee::DownwardDy(Hx, coefs_y, n_coefs_y, i, j, k, 0);
                Real const sx = cpml_stretched_derivative(cc, 0, Ez_stag[0], i, j, k, dt, true, dHy_dx, Pz(i,j,k,0));
                Real const sy = cpml_stretched_derivative(cc, 1, Ez_stag[1], i, j, k, dt, true, dHx_dy, Pz(i,j,k,1));
                Real const jz = has_j ? Jz(i,j,k) : 0._rt;
                Ez(i, j, k, 0) = alpha * Ez(i, j, k, 0) + beta * (sx - sy - jz);
            }
        );
    }
}
#endif // WARPX_DIM_RZ
