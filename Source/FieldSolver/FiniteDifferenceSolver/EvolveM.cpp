/* copyright
blank
*/

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"

#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

// update M field over one timestep

void FiniteDifferenceSolver::EvolveM ( std::unique_ptr<amrex::MultiFab>& Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt) {

    if (m_do_nodal) {

        EvolveMCartesian <CartesianNodalAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveMCartesian <CartesianYeeAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveMCartesian <CartesianCKCAlgorithm> ( Mfield, Bfield, dt );
    }
    else
    {
        amrex::Abort("Unknown algorithm");
    }
    
    template< typename T_Algo >
    void FiniteDifferenceSolver::EvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt )
    {
        static constexpr amrex::Real gamma = 1.759e-11;
        static constexpr amrex::Real alpha = 1e-4;
        static constexpr amrex::Real Ms = 1e4; 
        Real constexpr cons1 = - gamma;
        Real constexpr cons2 = -cons1*alpha/Ms;

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isvalid(); ++mfi)
        {
            // extract field data
            Array4<Real> const& Mx = Mfield[0]->array(mfi);
            Array4<Real> const& My = Mfield[1]->array(mfi);
            Array4<Real> const& Mz = Mfield[2]->array(mfi);
            Array4<Real> const& Bx = Bfield[0]->array(mfi);
            Array4<Real> const& By = Bfield[1]->array(mfi);
            Array4<Real> const& Bz = Bfield[2]->array(mfi);

            // extract stencil coefficients
            Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // extract tileboxes for which to loop
            Box const& tbx = mfi.tilebox(Mfield[0]->ixType().toIntVect());
            Box const& tby = mfi.tilebox(Mfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Mfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Mx(i, j, k) += dt * cons1 * ( My(i, j, k) * Bz(i, j, k) - Mz(i, j, k) * By(i, j, k))
                + dt * cons2 * ( My(i, j, k) * (Mx(i, j, k) * By(i, j, k) - My(i, j, k) * Bx(i, j, k))
                - Mz(i, j, k) * ( Mz(i, j, k) * Bx(i, j, k) - Mx(i, j, k) * Bz(i, j, k)));
            }

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                My(i, j, k) += dt * cons1 * ( Mz(i, j, k) * Bx(i, j, k) - Mx(i, j, k) * Bz(i, j, k))
                + dt * cons2 * ( Mz(i, j, k) * (My(i, j, k) * Bz(i, j, k) - Mz(i, j, k) * By(i, j, k))
                - Mx(i, j, k) * ( Mx(i, j, k) * By(i, j, k) - My(i, j, k) * Bx(i, j, k)));
            }

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Mz(i, j, k) += dt * cons1 * ( Mx(i, j, k) * By(i, j, k) - My(i, j, k) * Bx(i, j, k))
                + dt * cons2 * ( Mx(i, j, k) * ( Mz(i, j, k) * Bx(i, j, k) - Mx(i, j, k) * Bz(i, j, k))
                - My(i, j, k) * ( My(i, j, k) * Bz(i, j, k) - Mz(i, j, k) * By(i, j, k)));
            }
            );
        }
    }
}