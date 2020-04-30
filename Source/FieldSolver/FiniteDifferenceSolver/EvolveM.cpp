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

void FiniteDifferenceSolver::EvolveM ( 
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Mfield, // Mfield contains three components MultiFab
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt) {

    /* if (m_do_nodal) {

        EvolveMCartesian <CartesianNodalAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveMCartesian <CartesianYeeAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveMCartesian <CartesianCKCAlgorithm> ( Mfield, Bfield, dt );
    }
    else
    {
        amrex::Abort("Unknown algorithm");
    } */

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        EvolveMCartesian <CartesianYeeAlgorithm> (Mfield, Bfield, dt);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function EvolveM

    template<typename T_Algo>
    void FiniteDifferenceSolver::EvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield, 
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt )
    {
        static constexpr amrex::Real gamma = 1.759e-11;
        static constexpr amrex::Real alpha = 1e-4;
        static constexpr amrex::Real Ms = 1e4; 
        Real constexpr cons1 = -gamma;
        Real constexpr cons2 = -cons1*alpha/Ms;
        
        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
            // extract field data
            Array4<Real> const& Mx = Mfield[0]->array(mfi); // note Mx are x,y,z components at |_x faces
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
            Box const& tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const& tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on Mx(i,j,k, 0:2) we have direct access to Mx(i,j,k,0:2) and Bx(i,j,k)
              // By and Bz can be acquired by interpolation
              Real By_xtemp = 0.25*(By(i,j,k)+By(i,j+1,k)+By(i-1,j,k)+By(i-1,j+1,k));
              Real Bz_xtemp = 0.25*(Bz(i,j,k)+Bz(i-1,j,k)+Bz(i,j,k+1)+Bz(i-1,j,k+1));

              // now you have access to use Mx(i,j,k,0) Mx(i,j,k,1), Mx(i,j,k,2), Bx(i,j,k), By, Bz on the RHS of these update lines below

              // x component on x-faces of grid                                   
              Mx(i, j, k, 0) += dt * cons1 * ( Mx(i, j, k, 1) * Bz_xtemp - Mx(i, j, k, 2) * By_xtemp)
                + dt * cons2 * ( Mx(i, j, k, 1) * (Mx(i, j, k, 0) * By_xtemp - Mx(i, j, k, 1) * Bx(i, j, k))
                - Mx(i, j, k, 2) * ( Mx(i, j, k, 2) * Bx(i, j, k) - Mx(i, j, k, 0) * Bz_xtemp));

              // y component on x-faces of grid
              Mx(i, j, k, 1) += dt * cons1 * ( Mx(i, j, k, 2) * Bx(i, j, k) - Mx(i, j, k, 0) * Bz_xtemp)
                + dt * cons2 * ( Mx(i, j, k, 2) * (Mx(i, j, k, 1) * Bz_xtemp - Mx(i, j, k, 2) * By_xtemp)
                - Mx(i, j, k, 0) * ( Mx(i, j, k, 0) * By_xtemp - Mx(i, j, k, 1) * Bx(i, j, k)));

              // z component on x-faces of grid
              Mx(i, j, k, 2) += dt * cons1 * ( Mx(i, j, k, 0) * By_xtemp - Mx(i, j, k, 1) * Bx(i, j, k))
                + dt * cons2 * ( Mx(i, j, k, 0) * ( Mx(i, j, k, 2) * Bx(i, j, k) - Mx(i, j, k, 0) * Bz_xtemp)
                - Mx(i, j, k, 1) * ( Mx(i, j, k, 1) * Bz_xtemp - Mx(i, j, k, 2) * By_xtemp));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on My(i,j,k,0:2) we have direct access to My(i,j,k,0:2) and By(i,j,k)
              Real Bx_ytemp = 0.25*(Bx(i,j,k)+Bx(i+1,j,k)+Bx(i,j-1,k)+Bx(i+1,j-1,k));
              Real Bz_ytemp = 0.25*(Bz(i,j,k)+Bz(i,j,k+1)+Bz(i,j-1,k)+Bz(i,j-1,k+1));

              // x component on y-faces of grid                                   
              My(i, j, k, 0) += dt * cons1 * ( My(i, j, k, 1) * Bz_ytemp - My(i, j, k, 2) * By(i, j, k))
                + dt * cons2 * ( My(i, j, k, 1) * (My(i, j, k, 0) * By(i, j, k) - My(i, j, k, 1) * Bx(i, j, k))
                - My(i, j, k, 2) * ( My(i, j, k, 2) * Bx_ytemp - My(i, j, k, 0) * Bz_ytemp));

              // y component on y-faces of grid
              My(i, j, k, 1) += dt * cons1 * ( My(i, j, k, 2) * Bx_ytemp - My(i, j, k, 0) * Bz_ytemp)
                + dt * cons2 * ( My(i, j, k, 2) * (My(i, j, k, 1) * Bz_ytemp - My(i, j, k, 2) * By(i, j, k))
                - My(i, j, k, 0) * ( My(i, j, k, 0) * By(i, j, k) - My(i, j, k, 1) * Bx_ytemp));

              // z component on y-faces of grid
              My(i, j, k, 2) += dt * cons1 * ( My(i, j, k, 0) * By(i, j, k) - My(i, j, k, 1) * Bx_ytemp)
                + dt * cons2 * ( My(i, j, k, 0) * ( My(i, j, k, 2) * Bx_ytemp - My(i, j, k, 0) * Bz_ytemp)
                - My(i, j, k, 1) * ( My(i, j, k, 1) * Bz_ytemp - My(i, j, k, 2) * By(i, j, k)));   
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on Mz(i,j,k,0:2) we have direct access to Mz(i,j,k,0:2) and Bz(i,j,k)
              Real Bx_ztemp = 0.25*(Bx(i,j,k)+Bx(i+1,j,k)+Bx(i+1,j,k-1)+Bx(i,j,k-1));
              Real By_ztemp = 0.25*(By(i,j,k)+By(i,j,k-1)+By(i,j+1,k)+By(i,j+1,k-1));

              // x component on z-faces of grid                                   
              Mz(i, j, k, 0) += dt * cons1 * ( Mz(i, j, k, 1) * Bz(i, j, k) - Mz(i, j, k, 2) * By_ztemp)
                + dt * cons2 * ( Mz(i, j, k, 1) * (Mz(i, j, k, 0) * By_ztemp - Mz(i, j, k, 1) * Bx(i, j, k))
                - Mz(i, j, k, 2) * ( Mz(i, j, k, 2) * Bx_ztemp - Mz(i, j, k, 0) * Bz(i, j, k)));

              // y component on z-faces of grid
              Mz(i, j, k, 1) += dt * cons1 * ( Mz(i, j, k, 2) * Bx_ztemp - Mz(i, j, k, 0) * Bz(i, j, k))
                + dt * cons2 * ( Mz(i, j, k, 2) * (Mz(i, j, k, 1) * Bz(i, j, k) - Mz(i, j, k, 2) * By_ztemp)
                - Mz(i, j, k, 0) * ( Mz(i, j, k, 0) * By_ztemp - Mz(i, j, k, 1) * Bx_ztemp));

              // z component on z-faces of grid
              Mz(i, j, k, 2) += dt * cons1 * ( Mz(i, j, k, 0) * By_ztemp - Mz(i, j, k, 1) * Bx_ztemp)
                + dt * cons2 * ( Mz(i, j, k, 0) * ( Mz(i, j, k, 2) * Bx_ztemp - Mz(i, j, k, 0) * Bz(i, j, k))
                - Mz(i, j, k, 1) * ( Mz(i, j, k, 1) * Bz(i, j, k) - My(i, j, k, 2) * By_ztemp)); 
            }
            );
        }
    }