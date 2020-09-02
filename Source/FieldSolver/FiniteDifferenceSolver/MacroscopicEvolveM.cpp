/* copyright
blank
*/

#include "WarpX.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"

#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

#ifdef WARPX_MAG_LLG
// update M field over one timestep

void FiniteDifferenceSolver::MacroscopicEvolveM (
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Mfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties) {

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveMCartesian <CartesianYeeAlgorithm> (Mfield, H_biasfield, Bfield, dt, macroscopic_properties);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function EvolveM
#endif
#ifdef WARPX_MAG_LLG
    template<typename T_Algo>
    void FiniteDifferenceSolver::MacroscopicEvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties )
    {

        auto& warpx = WarpX::GetInstance();
        int coupling = warpx.mag_LLG_coupling;

        // build temporary vector<multifab,3> Mfield_prev
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_prev; // Mfield data in previous step
        for (int i = 0; i < 3; i++){
            Mfield_prev[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
            MultiFab::Copy(*Mfield_prev[i],*Mfield[i],0,0,3,Mfield[i]->nGrow());
        }

        // obtain the maximum relative amount we let M deviate from Ms before aborting
        amrex::Real mag_normalized_error = macroscopic_properties->getmag_normalized_error();

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
            auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
            auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
            auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
            // extract material properties
            Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
            Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
            Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);

            // extract field data
            Array4<Real> const& M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
            Array4<Real> const& M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
            Array4<Real> const& M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces
            Array4<Real> const& M_xface_prev = Mfield_prev[0]->array(mfi); // note M_xface_prev include x,y,z components at |_x faces
            Array4<Real> const& M_yface_prev = Mfield_prev[1]->array(mfi); // note M_yface_prev include x,y,z components at |_y faces
            Array4<Real> const& M_zface_prev = Mfield_prev[2]->array(mfi); // note M_zface_prev include x,y,z components at |_z faces
            Array4<Real> const& Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
            Array4<Real> const& Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
            Array4<Real> const& Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
            Array4<Real> const& Bx = Bfield[0]->array(mfi); // Bx is the x component at |_x faces
            Array4<Real> const& By = Bfield[1]->array(mfi); // By is the y component at |_y faces
            Array4<Real> const& Bz = Bfield[2]->array(mfi); // Bz is the z component at |_z faces

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

              // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
              // Hy and Hz can be acquired by interpolation

              // H_bias
              Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Hx_bias);
              Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), Hy_bias);
              Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Hz_bias);
              if (coupling == 1) {
                  // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                  // H_maxwell
                  Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Bx, M_xface);
                  Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), By, M_xface);
                  Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Bz, M_xface);
              }

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation. The IntVect is (1,0,0) to interpolate values to the x-face.
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_gamma_arr);
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

              // now you have access to use M_xface(i,j,k,0) M_xface(i,j,k,1), M_xface(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below
              // x component on x-faces of grid
              M_xface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_xface_prev(i, j, k, 1) * Hz_eff - M_xface_prev(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( M_xface_prev(i, j, k, 1) * (M_xface_prev(i, j, k, 0) * Hy_eff - M_xface_prev(i, j, k, 1) * Hx_eff)
                - M_xface_prev(i, j, k, 2) * ( M_xface_prev(i, j, k, 2) * Hx_eff - M_xface_prev(i, j, k, 0) * Hz_eff));

              // y component on x-faces of grid
              M_xface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_xface_prev(i, j, k, 2) * Hx_eff - M_xface_prev(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( M_xface_prev(i, j, k, 2) * (M_xface_prev(i, j, k, 1) * Hz_eff - M_xface_prev(i, j, k, 2) * Hy_eff)
                - M_xface_prev(i, j, k, 0) * ( M_xface_prev(i, j, k, 0) * Hy_eff - M_xface_prev(i, j, k, 1) * Hx_eff));

              // z component on x-faces of grid
              M_xface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_xface_prev(i, j, k, 0) * Hy_eff - M_xface_prev(i, j, k, 1) * Hx_eff)
                + dt * Gil_damp * ( M_xface_prev(i, j, k, 0) * (M_xface_prev(i, j, k, 2) * Hx_eff - M_xface_prev(i, j, k, 0) * Hz_eff)
                - M_xface_prev(i, j, k, 1) * ( M_xface_prev(i, j, k, 1) * Hz_eff - M_xface_prev(i, j, k, 2) * Hy_eff));


              // temporary normalized magnitude of M_xface field at the fixed point
              // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
              amrex::Real mag_normalized = std::sqrt( std::pow(M_xface(i, j, k, 0),2.0) + std::pow(M_xface(i, j, k, 1),2.0) +
                      std::pow(M_xface(i, j, k, 2),2.0) ) / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

              // check the normalized error
              if ( amrex::Math::abs(1._rt-mag_normalized) > mag_normalized_error ){
                  printf("i = %d, j=%d, k=%d\n", i, j, k);
                  printf("mag_normalized = %f, mag_normalized_error=%f\n", mag_normalized, mag_normalized_error);
                  amrex::Abort("Exceed the normalized error of the M_xface field");
              }
              // normalize the M_xface field
              M_xface(i,j,k,0) /= mag_normalized;
              M_xface(i,j,k,1) /= mag_normalized;
              M_xface(i,j,k,2) /= mag_normalized;
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
              // Hy and Hz can be acquired by interpolation

              // H_bias
              Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Hx_bias);
              Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), Hy_bias);
              Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Hz_bias);
              if (coupling == 1) {
                  // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                  // H_maxwell
                  Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Bx, M_yface);
                  Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), By, M_yface);
                  Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Bz, M_yface);
              }

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation. The IntVect is (0,1,0) to interpolate values to the y-face.
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_gamma_arr);
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);

              // x component on y-faces of grid
              M_yface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_yface_prev(i, j, k, 1) * Hz_eff - M_yface_prev(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( M_yface_prev(i, j, k, 1) * (M_yface_prev(i, j, k, 0) * Hy_eff - M_yface_prev(i, j, k, 1) * Hx_eff)
                - M_yface_prev(i, j, k, 2) * ( M_yface_prev(i, j, k, 2) * Hx_eff - M_yface_prev(i, j, k, 0) * Hz_eff));

              // y component on y-faces of grid
              M_yface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_yface_prev(i, j, k, 2) * Hx_eff - M_yface_prev(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( M_yface_prev(i, j, k, 2) * (M_yface_prev(i, j, k, 1) * Hz_eff - M_yface_prev(i, j, k, 2) * Hy_eff)
                - M_yface_prev(i, j, k, 0) * ( M_yface_prev(i, j, k, 0) * Hy_eff - M_yface_prev(i, j, k, 1) * Hx_eff));

              // z component on y-faces of grid
              M_yface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_yface_prev(i, j, k, 0) * Hy_eff - M_yface_prev(i, j, k, 1) * Hx_eff)
                + dt * Gil_damp * ( M_yface_prev(i, j, k, 0) * (M_yface_prev(i, j, k, 2) * Hx_eff - M_yface_prev(i, j, k, 0) * Hz_eff)
                - M_yface_prev(i, j, k, 1) * ( M_yface_prev(i, j, k, 1) * Hz_eff - M_yface_prev(i, j, k, 2) * Hy_eff));


              // temporary normalized magnitude of M_yface field at the fixed point
              // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
              amrex::Real mag_normalized = std::sqrt( std::pow(M_yface(i, j, k, 0),2.0) + std::pow(M_yface(i, j, k, 1),2.0) +
                      std::pow(M_yface(i, j, k, 2),2.0) ) / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);

              // check the normalized error
              if ( amrex::Math::abs(1._rt-mag_normalized) > mag_normalized_error ){
                 printf("i = %d, j=%d, k=%d\n", i, j, k);
                 printf("mag_normalized = %f, mag_normalized_error=%f\n",mag_normalized, mag_normalized_error);
                 amrex::Abort("Exceed the normalized error of the M_yface field");
              }
              // normalize the M_yface field
              M_yface(i,j,k,0) /= mag_normalized;
              M_yface(i,j,k,1) /= mag_normalized;
              M_yface(i,j,k,2) /= mag_normalized;
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
              // Hy and Hz can be acquired by interpolation

              // H_bias
              Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Hx_bias);
              Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), Hy_bias);
              Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Hz_bias);

              if (coupling == 1) {
                  // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                  // H_maxwell
                  Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Bx, M_zface);
                  Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), By, M_zface);
                  Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Bz, M_zface);
              }

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation. The IntVect is (0,0,1) to interpolate values to the z-face.
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_gamma_arr);
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);

              // x component on z-faces of grid
              M_zface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_zface_prev(i, j, k, 1) * Hz_eff - M_zface_prev(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( M_zface_prev(i, j, k, 1) * (M_zface_prev(i, j, k, 0) * Hy_eff - M_zface_prev(i, j, k, 1) * Hx_eff)
                - M_zface_prev(i, j, k, 2) * ( M_zface_prev(i, j, k, 2) * Hx_eff - M_zface_prev(i, j, k, 0) * Hz_eff));

              // y component on z-faces of grid
              M_zface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_zface_prev(i, j, k, 2) * Hx_eff - M_zface_prev(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( M_zface_prev(i, j, k, 2) * (M_zface_prev(i, j, k, 1) * Hz_eff - M_zface_prev(i, j, k, 2) * Hy_eff)
                - M_zface_prev(i, j, k, 0) * ( M_zface_prev(i, j, k, 0) * Hy_eff - M_zface_prev(i, j, k, 1) * Hx_eff));

              // z component on z-faces of grid
              M_zface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * ( M_zface_prev(i, j, k, 0) * Hy_eff - M_zface_prev(i, j, k, 1) * Hx_eff)
                + dt * Gil_damp * ( M_zface_prev(i, j, k, 0) * (M_zface_prev(i, j, k, 2) * Hx_eff - M_zface_prev(i, j, k, 0) * Hz_eff)
                - M_zface_prev(i, j, k, 1) * ( M_zface_prev(i, j, k, 1) * Hz_eff - M_yface_prev(i, j, k, 2) * Hy_eff));

              // temporary normalized magnitude of M_zface field at the fixed point
              // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
              amrex::Real mag_normalized = std::sqrt( std::pow(M_zface(i, j, k, 0),2.0_rt) + std::pow(M_zface(i, j, k, 1),2.0_rt) +
                      std::pow(M_zface(i, j, k, 2),2.0_rt) ) / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);

              // check the normalized error
              if ( amrex::Math::abs(1.-mag_normalized) > mag_normalized_error ){
                 printf("i = %d, j=%d, k=%d\n", i, j, k);
                 printf("mag_normalized = %f, mag_normalized_error=%f\n", mag_normalized, mag_normalized_error);
                 amrex::Abort("Exceed the normalized error of the M_zface field");
              }
              // normalize the M_zface field
              M_zface(i,j,k,0) /= mag_normalized;
              M_zface(i,j,k,1) /= mag_normalized;
              M_zface(i,j,k,2) /= mag_normalized;});
        }
    }
#endif
