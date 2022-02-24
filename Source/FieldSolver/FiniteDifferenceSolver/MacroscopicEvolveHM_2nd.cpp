/* copyright
blank
*/

#include "WarpX.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"

#include "Utils/WarpXConst.H"
#include "Utils/CoarsenIO.H"
#include "Utils/WarpXUtil.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update H and M fields with iterative correction, over one timestep
 */

#ifndef WARPX_DIM_RZ
#ifdef WARPX_MAG_LLG

void FiniteDifferenceSolver::MacroscopicEvolveHM_2nd(
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    int lev,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield, // Mfield contains three components MultiFab
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties) {

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee){
        MacroscopicEvolveHMCartesian_2nd<CartesianYeeAlgorithm>(lev, Mfield, Hfield, Bfield, H_biasfield, Efield, dt, macroscopic_properties);
    } else {
        amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
} // closes function MacroscopicEvolveHM_2nd
#endif
#ifdef WARPX_MAG_LLG
template <typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveHMCartesian_2nd(
    int lev,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties) {

    // obtain the maximum relative amount we let M deviate from Ms before aborting
    amrex::Real mag_normalized_error = macroscopic_properties->getmag_normalized_error();

    auto &warpx = WarpX::GetInstance();
    int coupling = warpx.mag_LLG_coupling;
    int M_normalization = warpx.mag_M_normalization;
    int mag_exchange_coupling = warpx.mag_LLG_exchange_coupling;
    int mag_anisotropy_coupling = warpx.mag_LLG_anisotropy_coupling;

    // build temporary vector<multifab,3> Mfield_prev, Mfield_error, a_temp, a_temp_static, b_temp_static
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Hfield_old;    // H^(old_time) before the current time step
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_old;    // M^(old_time) before the current time step
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_prev;   // M^(new_time) of the (r-1)th iteration
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_error;  // The error of the M field between the two consecutive iterations
    std::array<std::unique_ptr<amrex::MultiFab>, 3> a_temp;        // right-hand side of vector a, see the documentation
    std::array<std::unique_ptr<amrex::MultiFab>, 3> a_temp_static; // α M^(old_time)/|M| in the right-hand side of vector a, see the documentation
    std::array<std::unique_ptr<amrex::MultiFab>, 3> b_temp_static; // right-hand side of vector b, see the documentation

    amrex::GpuArray<int, 3> const& mu_stag  = macroscopic_properties->mu_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag  = macroscopic_properties->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag  = macroscopic_properties->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag  = macroscopic_properties->Bz_IndexType;
    amrex::GpuArray<int, 3> const& Hx_stag  = macroscopic_properties->Hx_IndexType;
    amrex::GpuArray<int, 3> const& Hy_stag  = macroscopic_properties->Hy_IndexType;
    amrex::GpuArray<int, 3> const& Hz_stag  = macroscopic_properties->Hz_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr = macroscopic_properties->macro_cr_ratio;
    amrex::GpuArray<amrex::Real, 3> const& anisotropy_axis = macroscopic_properties->mag_LLG_anisotropy_axis;

    // Initialize Hfield_old (H^(old_time)), Mfield_old (M^(old_time)), Mfield_prev (M^[(new_time),r-1]), Mfield_error
    for (int i = 0; i < 3; i++){
        Hfield_old[i].reset(new MultiFab(Hfield[i]->boxArray(), Hfield[i]->DistributionMap(), 1, Hfield[i]->nGrow()));
        Mfield_old[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        Mfield_prev[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        Mfield_error[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        Mfield_error[i]->setVal(0.); // reset Mfield_error to zero
        MultiFab::Copy(*Hfield_old[i], *Hfield[i], 0, 0, 1, Hfield[i]->nGrow());
        MultiFab::Copy(*Mfield_old[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
        MultiFab::Copy(*Mfield_prev[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
    }
    // initialize a_temp, a_temp_static, b_temp_static
    for (int i = 0; i < 3; i++){
        a_temp[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        a_temp_static[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        b_temp_static[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
    }

    amrex::MultiFab& mu_mf = macroscopic_properties->getmu_mf();

    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(*a_temp_static[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        auto& mag_Ms_xface_mf = macroscopic_properties->getmag_Ms_mf(0);
        auto& mag_Ms_yface_mf = macroscopic_properties->getmag_Ms_mf(1);
        auto& mag_Ms_zface_mf = macroscopic_properties->getmag_Ms_mf(2);
        auto& mag_alpha_xface_mf = macroscopic_properties->getmag_alpha_mf(0);
        auto& mag_alpha_yface_mf = macroscopic_properties->getmag_alpha_mf(1);
        auto& mag_alpha_zface_mf = macroscopic_properties->getmag_alpha_mf(2);
        auto& mag_gamma_xface_mf = macroscopic_properties->getmag_gamma_mf(0);
        auto& mag_gamma_yface_mf = macroscopic_properties->getmag_gamma_mf(1);
        auto& mag_gamma_zface_mf = macroscopic_properties->getmag_gamma_mf(2);
        auto& mag_exchange_xface_mf = macroscopic_properties->getmag_exchange_mf(0);
        auto& mag_exchange_yface_mf = macroscopic_properties->getmag_exchange_mf(1);
        auto& mag_exchange_zface_mf = macroscopic_properties->getmag_exchange_mf(2);
        auto& mag_anisotropy_xface_mf = macroscopic_properties->getmag_anisotropy_mf(0);
        auto& mag_anisotropy_yface_mf = macroscopic_properties->getmag_anisotropy_mf(1);
        auto& mag_anisotropy_zface_mf = macroscopic_properties->getmag_anisotropy_mf(2);

        // extract material properties
        Array4<Real> const& mag_Ms_xface_arr = mag_Ms_xface_mf.array(mfi);
        Array4<Real> const& mag_Ms_yface_arr = mag_Ms_yface_mf.array(mfi);
        Array4<Real> const& mag_Ms_zface_arr = mag_Ms_zface_mf.array(mfi);
        Array4<Real> const& mag_alpha_xface_arr = mag_alpha_xface_mf.array(mfi);
        Array4<Real> const& mag_alpha_yface_arr = mag_alpha_yface_mf.array(mfi);
        Array4<Real> const& mag_alpha_zface_arr = mag_alpha_zface_mf.array(mfi);
        Array4<Real> const& mag_gamma_xface_arr = mag_gamma_xface_mf.array(mfi);
        Array4<Real> const& mag_gamma_yface_arr = mag_gamma_yface_mf.array(mfi);
        Array4<Real> const& mag_gamma_zface_arr = mag_gamma_zface_mf.array(mfi);
        Array4<Real> const& mag_exchange_xface_arr = mag_exchange_xface_mf.array(mfi);
        Array4<Real> const& mag_exchange_yface_arr = mag_exchange_yface_mf.array(mfi);
        Array4<Real> const& mag_exchange_zface_arr = mag_exchange_zface_mf.array(mfi);
        Array4<Real> const& mag_anisotropy_xface_arr = mag_anisotropy_xface_mf.array(mfi);
        Array4<Real> const& mag_anisotropy_yface_arr = mag_anisotropy_yface_mf.array(mfi);
        Array4<Real> const& mag_anisotropy_zface_arr = mag_anisotropy_zface_mf.array(mfi);

        // extract field data
        Array4<Real> const &M_xface = Mfield[0]->array(mfi);      // note M_xface include x,y,z components at |_x faces
        Array4<Real> const &M_yface = Mfield[1]->array(mfi);      // note M_yface include x,y,z components at |_y faces
        Array4<Real> const &M_zface = Mfield[2]->array(mfi);      // note M_zface include x,y,z components at |_z faces
        Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
        Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
        Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
        Array4<Real> const &Hx_old = Hfield_old[0]->array(mfi);   // Hx_old is the x component at |_x faces
        Array4<Real> const &Hy_old = Hfield_old[1]->array(mfi);   // Hy_old is the y component at |_y faces
        Array4<Real> const &Hz_old = Hfield_old[2]->array(mfi);   // Hz_old is the z component at |_z faces

        // extract field data of a_temp_static and b_temp_static
        Array4<Real> const &a_temp_static_xface = a_temp_static[0]->array(mfi);
        Array4<Real> const &a_temp_static_yface = a_temp_static[1]->array(mfi);
        Array4<Real> const &a_temp_static_zface = a_temp_static[2]->array(mfi);
        Array4<Real> const &b_temp_static_xface = b_temp_static[0]->array(mfi);
        Array4<Real> const &b_temp_static_yface = b_temp_static[1]->array(mfi);
        Array4<Real> const &b_temp_static_zface = b_temp_static[2]->array(mfi);

        // extract tileboxes for which to loop
        amrex::IntVect Mxface_stag = Mfield[0]->ixType().toIntVect();
        amrex::IntVect Myface_stag = Mfield[1]->ixType().toIntVect();
        amrex::IntVect Mzface_stag = Mfield[2]->ixType().toIntVect();
        Box const &tbx = mfi.tilebox(Mxface_stag); /* just define which grid type */
        Box const &tby = mfi.tilebox(Myface_stag);
        Box const &tbz = mfi.tilebox(Mzface_stag);

        // Extract stencil coefficients for calculating the exchange field H_exchange and the anisotropy field H_anisotropy
        amrex::Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        amrex::Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        amrex::Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // loop over cells and update fields
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (mag_Ms_xface_arr(i,j,k) > 0._rt){

                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz_old);
                    }

                    if (mag_exchange_coupling == 1){

                        if (mag_exchange_xface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_xface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_xface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_xface_arr(i,j,k) / mag_Ms_xface_arr(i,j,k);

                        amrex::Real Ms_lo_x = mag_Ms_xface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = mag_Ms_xface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = mag_Ms_xface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = mag_Ms_xface_arr(i, j+1, k);
                        amrex::Real Ms_lo_z = mag_Ms_xface_arr(i, j, k-1);
                        amrex::Real Ms_hi_z = mag_Ms_xface_arr(i, j, k+1);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 0); //Last argument is nodality -- xface = 0
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 0); //Last argument is nodality -- xface = 0
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 0); //Last argument is nodality -- xface = 0
                    }

                    if (mag_anisotropy_coupling == 1){

                        if (mag_anisotropy_xface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_xface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_xface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_xface_arr(i,j,k) / mag_Ms_xface_arr(i,j,k);
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_xface_arr(i,j,k);
                    // a_temp_static_coeff does not change in the current step for SATURATED materials; but it does change for UNSATURATED ones
                    amrex::Real a_temp_static_coeff = mag_alpha_xface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_xface_arr(i,j,k)) / 2._rt;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_xface
                        // all components on x-faces of grid
                        a_temp_static_xface(i, j, k, comp) = a_temp_static_coeff * M_xface(i, j, k, comp);
                    }

                    // calculate b_temp_static_xface
                    // x component on x-faces of grid
                    b_temp_static_xface(i, j, k, 0) = M_xface(i, j, k, 0) + dt * b_temp_static_coeff * (M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff);

                    // y component on x-faces of grid
                    b_temp_static_xface(i, j, k, 1) = M_xface(i, j, k, 1) + dt * b_temp_static_coeff * (M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff);

                    // z component on x-faces of grid
                    b_temp_static_xface(i, j, k, 2) = M_xface(i, j, k, 2) + dt * b_temp_static_coeff * (M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff);
                }
            });

        amrex::ParallelFor(tby,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (mag_Ms_yface_arr(i,j,k) > 0._rt){

                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz_old);
                    }

                    if (mag_exchange_coupling == 1){

                        if (mag_exchange_yface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_yface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_yface_arr(i,j,k) / mag_Ms_yface_arr(i,j,k);

                        amrex::Real Ms_lo_x = mag_Ms_yface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = mag_Ms_yface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = mag_Ms_yface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = mag_Ms_yface_arr(i, j+1, k);
                        amrex::Real Ms_lo_z = mag_Ms_yface_arr(i, j, k-1);
                        amrex::Real Ms_hi_z = mag_Ms_yface_arr(i, j, k+1);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 1); //Last argument is nodality -- yface = 1
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 1); //Last argument is nodality -- yface = 1
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 1); //Last argument is nodality -- yface = 1
                    }

                    if (mag_anisotropy_coupling == 1){

                        if (mag_anisotropy_yface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_yface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_yface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_yface_arr(i,j,k) / mag_Ms_yface_arr(i,j,k);
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    // note the unsaturated case is less usefull in real devices
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_yface_arr(i,j,k);
                    amrex::Real a_temp_static_coeff = mag_alpha_yface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_yface_arr(i,j,k)) / 2._rt;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_yface
                        // all component on y-faces of grid
                        a_temp_static_yface(i, j, k, comp) = a_temp_static_coeff * M_yface(i, j, k, comp);
                    }

                    // calculate b_temp_static_yface
                    // x component on y-faces of grid
                    b_temp_static_yface(i, j, k, 0) = M_yface(i, j, k, 0) + dt * b_temp_static_coeff * (M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff);

                    // y component on y-faces of grid
                    b_temp_static_yface(i, j, k, 1) = M_yface(i, j, k, 1) + dt * b_temp_static_coeff * (M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff);

                    // z component on y-faces of grid
                    b_temp_static_yface(i, j, k, 2) = M_yface(i, j, k, 2) + dt * b_temp_static_coeff * (M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff);
                }
            });

        amrex::ParallelFor(tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (mag_Ms_zface_arr(i,j,k) > 0._rt){

                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz_old);
                    }

                    if (mag_exchange_coupling == 1){

                        if (mag_exchange_zface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_zface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_zface_arr(i,j,k) / mag_Ms_zface_arr(i,j,k);

                        amrex::Real Ms_lo_x = mag_Ms_zface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = mag_Ms_zface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = mag_Ms_zface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = mag_Ms_zface_arr(i, j+1, k);
                        amrex::Real Ms_lo_z = mag_Ms_zface_arr(i, j, k-1);
                        amrex::Real Ms_hi_z = mag_Ms_zface_arr(i, j, k+1);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 2); //Last argument is nodality -- zface = 2
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 2); //Last argument is nodality -- zface = 2
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 2); //Last argument is nodality -- zface = 2
                    }

                    if (mag_anisotropy_coupling == 1){

                        if (mag_anisotropy_zface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_zface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_zface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_zface_arr(i,j,k) / mag_Ms_zface_arr(i,j,k);
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_zface_arr(i,j,k);
                    amrex::Real a_temp_static_coeff = mag_alpha_zface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_zface_arr(i,j,k)) / 2._rt;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_zface
                        // all components on z-faces of grid
                        a_temp_static_zface(i, j, k, comp) = a_temp_static_coeff * M_zface(i, j, k, comp);
                    }

                    // calculate b_temp_static_zface
                    // x component on z-faces of grid
                    b_temp_static_zface(i, j, k, 0) = M_zface(i, j, k, 0) + dt * b_temp_static_coeff * (M_zface(i, j, k, 1) * Hz_eff - M_zface(i, j, k, 2) * Hy_eff);

                    // y component on z-faces of grid
                    b_temp_static_zface(i, j, k, 1) = M_zface(i, j, k, 1) + dt * b_temp_static_coeff * (M_zface(i, j, k, 2) * Hx_eff - M_zface(i, j, k, 0) * Hz_eff);

                    // z component on z-faces of grid
                    b_temp_static_zface(i, j, k, 2) = M_zface(i, j, k, 2) + dt * b_temp_static_coeff * (M_zface(i, j, k, 0) * Hy_eff - M_zface(i, j, k, 1) * Hx_eff);
                }
            });
    }

    // initialize M_max_iter, M_iter, M_tol, M_iter_error
    // maximum number of iterations allowed
    int M_max_iter = macroscopic_properties->getmag_max_iter();
    int M_iter = 0;
    // relative tolerance stopping criteria for 2nd-order iterative algorithm
    amrex::Real M_tol = macroscopic_properties->getmag_tol();
    int stop_iter = 0;

    // begin the iteration
    while (!stop_iter){

        warpx.FillBoundaryH(warpx.getngEB());

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

            auto& mag_Ms_xface_mf = macroscopic_properties->getmag_Ms_mf(0);
            auto& mag_Ms_yface_mf = macroscopic_properties->getmag_Ms_mf(1);
            auto& mag_Ms_zface_mf = macroscopic_properties->getmag_Ms_mf(2);
            auto& mag_alpha_xface_mf = macroscopic_properties->getmag_alpha_mf(0);
            auto& mag_alpha_yface_mf = macroscopic_properties->getmag_alpha_mf(1);
            auto& mag_alpha_zface_mf = macroscopic_properties->getmag_alpha_mf(2);
            auto& mag_gamma_xface_mf = macroscopic_properties->getmag_gamma_mf(0);
            auto& mag_gamma_yface_mf = macroscopic_properties->getmag_gamma_mf(1);
            auto& mag_gamma_zface_mf = macroscopic_properties->getmag_gamma_mf(2);
            auto& mag_exchange_xface_mf = macroscopic_properties->getmag_exchange_mf(0);
            auto& mag_exchange_yface_mf = macroscopic_properties->getmag_exchange_mf(1);
            auto& mag_exchange_zface_mf = macroscopic_properties->getmag_exchange_mf(2);
            auto& mag_anisotropy_xface_mf = macroscopic_properties->getmag_anisotropy_mf(0);
            auto& mag_anisotropy_yface_mf = macroscopic_properties->getmag_anisotropy_mf(1);
            auto& mag_anisotropy_zface_mf = macroscopic_properties->getmag_anisotropy_mf(2);

            // extract material properties
            Array4<Real> const& mag_Ms_xface_arr = mag_Ms_xface_mf.array(mfi);
            Array4<Real> const& mag_Ms_yface_arr = mag_Ms_yface_mf.array(mfi);
            Array4<Real> const& mag_Ms_zface_arr = mag_Ms_zface_mf.array(mfi);
            Array4<Real> const& mag_alpha_xface_arr = mag_alpha_xface_mf.array(mfi);
            Array4<Real> const& mag_alpha_yface_arr = mag_alpha_yface_mf.array(mfi);
            Array4<Real> const& mag_alpha_zface_arr = mag_alpha_zface_mf.array(mfi);
            Array4<Real> const& mag_gamma_xface_arr = mag_gamma_xface_mf.array(mfi);
            Array4<Real> const& mag_gamma_yface_arr = mag_gamma_yface_mf.array(mfi);
            Array4<Real> const& mag_gamma_zface_arr = mag_gamma_zface_mf.array(mfi);
            Array4<Real> const& mag_exchange_xface_arr = mag_exchange_xface_mf.array(mfi);
            Array4<Real> const& mag_exchange_yface_arr = mag_exchange_yface_mf.array(mfi);
            Array4<Real> const& mag_exchange_zface_arr = mag_exchange_zface_mf.array(mfi);
            Array4<Real> const& mag_anisotropy_xface_arr = mag_anisotropy_xface_mf.array(mfi);
            Array4<Real> const& mag_anisotropy_yface_arr = mag_anisotropy_yface_mf.array(mfi);
            Array4<Real> const& mag_anisotropy_zface_arr = mag_anisotropy_zface_mf.array(mfi);

            // extract field data
            Array4<Real> const &M_xface = Mfield[0]->array(mfi);      // note M_xface include x,y,z components at |_x faces
            Array4<Real> const &M_yface = Mfield[1]->array(mfi);      // note M_yface include x,y,z components at |_y faces
            Array4<Real> const &M_zface = Mfield[2]->array(mfi);      // note M_zface include x,y,z components at |_z faces
            Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
            Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
            Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
            Array4<Real> const &Hx = Hfield[0]->array(mfi);           // Hx is the x component at |_x faces
            Array4<Real> const &Hy = Hfield[1]->array(mfi);           // Hy is the y component at |_y faces
            Array4<Real> const &Hz = Hfield[2]->array(mfi);           // Hz is the z component at |_z faces

            // extract field data of Mfield_prev, Mfield_error, a_temp, a_temp_static, and b_temp_static
            Array4<Real> const &M_prev_xface = Mfield_prev[0]->array(mfi);
            Array4<Real> const &M_prev_yface = Mfield_prev[1]->array(mfi);
            Array4<Real> const &M_prev_zface = Mfield_prev[2]->array(mfi);
            Array4<Real> const &M_old_xface = Mfield_old[0]->array(mfi);
            Array4<Real> const &M_old_yface = Mfield_old[1]->array(mfi);
            Array4<Real> const &M_old_zface = Mfield_old[2]->array(mfi);
            Array4<Real> const &M_error_xface = Mfield_error[0]->array(mfi);
            Array4<Real> const &M_error_yface = Mfield_error[1]->array(mfi);
            Array4<Real> const &M_error_zface = Mfield_error[2]->array(mfi);
            Array4<Real> const &a_temp_xface = a_temp[0]->array(mfi);
            Array4<Real> const &a_temp_yface = a_temp[1]->array(mfi);
            Array4<Real> const &a_temp_zface = a_temp[2]->array(mfi);
            Array4<Real> const &a_temp_static_xface = a_temp_static[0]->array(mfi);
            Array4<Real> const &a_temp_static_yface = a_temp_static[1]->array(mfi);
            Array4<Real> const &a_temp_static_zface = a_temp_static[2]->array(mfi);
            Array4<Real> const &b_temp_static_xface = b_temp_static[0]->array(mfi);
            Array4<Real> const &b_temp_static_yface = b_temp_static[1]->array(mfi);
            Array4<Real> const &b_temp_static_zface = b_temp_static[2]->array(mfi);

            // extract tileboxes for which to loop
            amrex::IntVect Hxnodal = Hfield[0]->ixType().toIntVect();
            amrex::IntVect Hynodal = Hfield[1]->ixType().toIntVect();
            amrex::IntVect Hznodal = Hfield[2]->ixType().toIntVect();
            Box const &tbx = mfi.tilebox(Hxnodal); /* just define which grid type */
            Box const &tby = mfi.tilebox(Hynodal);
            Box const &tbz = mfi.tilebox(Hznodal);

            // Extract stencil coefficients for calculating the exchange field H_exchange and the anisotropy field H_anisotropy
            amrex::Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            amrex::Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            amrex::Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // loop over cells and update fields
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_xface_arr(i,j,k) > 0._rt){

                        // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hxnodal, Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hxnodal, Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hxnodal, Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hxnodal, Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hxnodal, Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hxnodal, Hz);
                        }

                        if (mag_exchange_coupling == 1){

                            if (mag_exchange_xface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_xface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_xface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_xface_arr(i,j,k) / mag_Ms_xface_arr(i,j,k);

                            amrex::Real Ms_lo_x = mag_Ms_xface_arr(i-1, j, k);
                            amrex::Real Ms_hi_x = mag_Ms_xface_arr(i+1, j, k);
                            amrex::Real Ms_lo_y = mag_Ms_xface_arr(i, j-1, k);
                            amrex::Real Ms_hi_y = mag_Ms_xface_arr(i, j+1, k);
                            amrex::Real Ms_lo_z = mag_Ms_xface_arr(i, j, k-1);
                            amrex::Real Ms_hi_z = mag_Ms_xface_arr(i, j, k+1);

                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 0); //Last argument is nodality -- xface = 0
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 0); //Last argument is nodality -- xface = 0
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 0); //Last argument is nodality -- xface = 0
                        }

                        if (mag_anisotropy_coupling == 1){

                            if (mag_anisotropy_xface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_xface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_xface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_xface_arr(i,j,k) / mag_Ms_xface_arr(i,j,k);
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_xface_arr(i,j,k)) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_xface
                            // all components on x-faces of grid
                            a_temp_xface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_xface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_xface(i, j, k, comp)
                                                                                     + 0.5 * mag_alpha_xface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) * M_old_xface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_xface from a and b using the updateM_field
                            // all components on x-faces of grid
                            M_xface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_xface, b_temp_static_xface);
                        }

                        // temporary normalized magnitude of M_xface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / mag_Ms_xface_arr(i,j,k);
                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_xface field");
                            }
                            // normalize the M_xface field
                            M_xface(i, j, k, 0) /= M_magnitude_normalized;
                            M_xface(i, j, k, 1) /= M_magnitude_normalized;
                            M_xface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > (1._rt + mag_normalized_error)){
                                amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error){
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_xface
                        // x,y,z component on M-error on x-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_error_xface(i, j, k, icomp) = amrex::Math::abs((M_xface(i, j, k, icomp) - M_prev_xface(i, j, k, icomp))) / mag_Ms_xface_arr(i,j,k);
                        }
                    }
                });

            amrex::ParallelFor(tby,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_yface_arr(i,j,k) > 0._rt){

                        // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hynodal, Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hynodal, Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hynodal, Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hynodal, Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hynodal, Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hynodal, Hz);
                        }

                        if (mag_exchange_coupling == 1){

                            if (mag_exchange_yface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_yface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_yface_arr(i,j,k) / mag_Ms_yface_arr(i,j,k);

                            amrex::Real Ms_lo_x = mag_Ms_yface_arr(i-1, j, k);
                            amrex::Real Ms_hi_x = mag_Ms_yface_arr(i+1, j, k);
                            amrex::Real Ms_lo_y = mag_Ms_yface_arr(i, j-1, k);
                            amrex::Real Ms_hi_y = mag_Ms_yface_arr(i, j+1, k);
                            amrex::Real Ms_lo_z = mag_Ms_yface_arr(i, j, k-1);
                            amrex::Real Ms_hi_z = mag_Ms_yface_arr(i, j, k+1);

                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 1); //Last argument is nodality -- yface = 1
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 1); //Last argument is nodality -- yface = 1
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 1); //Last argument is nodality -- yface = 1
                        }

                        if (mag_anisotropy_coupling == 1){

                            if (mag_anisotropy_yface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_yface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_yface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_yface_arr(i,j,k) / mag_Ms_yface_arr(i,j,k);
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_yface_arr(i,j,k)) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_yface
                            // all components on y-faces of grid
                            a_temp_yface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_yface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_yface(i, j, k, comp)
                                                                                     + 0.5 * mag_alpha_yface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) * M_old_yface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_yface from a and b using the updateM_field
                            // all components on y-faces of grid
                            M_yface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_yface, b_temp_static_yface);
                        }

                        // temporary normalized magnitude of M_yface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / mag_Ms_yface_arr(i,j,k);

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_yface field");
                            }
                            // normalize the M_yface field
                            M_yface(i, j, k, 0) /= M_magnitude_normalized;
                            M_yface(i, j, k, 1) /= M_magnitude_normalized;
                            M_yface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > 1._rt + mag_normalized_error){
                                amrex::Abort("Caution: Unsaturated material has M_yface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error){
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_yface
                        // x,y,z component on y-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_error_yface(i, j, k, icomp) = amrex::Math::abs((M_yface(i, j, k, icomp) - M_prev_yface(i, j, k, icomp))) / mag_Ms_yface_arr(i,j,k);
                        }
                    }
                });

            amrex::ParallelFor(tbz,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_zface_arr(i,j,k) > 0._rt){

                        // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hznodal, Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hznodal, Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hznodal, Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hxnodal, Hznodal, Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hynodal, Hznodal, Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Hznodal, Hznodal, Hz);
                        }

                        if (mag_exchange_coupling == 1){

                            if (mag_exchange_zface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_zface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_zface_arr(i,j,k) / mag_Ms_zface_arr(i,j,k);

                            amrex::Real Ms_lo_x = mag_Ms_zface_arr(i-1, j, k);
                            amrex::Real Ms_hi_x = mag_Ms_zface_arr(i+1, j, k);
                            amrex::Real Ms_lo_y = mag_Ms_zface_arr(i, j-1, k);
                            amrex::Real Ms_hi_y = mag_Ms_zface_arr(i, j+1, k);
                            amrex::Real Ms_lo_z = mag_Ms_zface_arr(i, j, k-1);
                            amrex::Real Ms_hi_z = mag_Ms_zface_arr(i, j, k+1);

                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 2); //Last argument is nodality -- zface = 2
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 2); //Last argument is nodality -- zface = 2
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_prev_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 2); //Last argument is nodality -- zface = 2
                        }

                        if (mag_anisotropy_coupling == 1){

                            if (mag_anisotropy_zface_arr(i,j,k) == 0._rt) amrex::Abort("The mag_anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_zface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_zface_arr(i,j,k) / PhysConst::mu0 / mag_Ms_zface_arr(i,j,k) / mag_Ms_zface_arr(i,j,k);
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_zface_arr(i,j,k)) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_zface
                            // all components on z-faces of grid
                            a_temp_zface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_zface(i, j, k, comp))
                                                                              : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_zface(i, j, k, comp)
                                                                                  + 0.5 * mag_alpha_zface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) * M_old_zface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_zface from a and b using the updateM_field
                            // all components on z-faces of grid
                            M_zface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_zface, b_temp_static_zface);
                        }

                        // temporary normalized magnitude of M_zface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / mag_Ms_zface_arr(i,j,k);

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_zface field");
                            }
                            // normalize the M_zface field
                            M_zface(i, j, k, 0) /= M_magnitude_normalized;
                            M_zface(i, j, k, 1) /= M_magnitude_normalized;
                            M_zface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > 1._rt + mag_normalized_error){
                                amrex::Abort("Caution: Unsaturated material has M_zface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error){
                                // normalize the M_zface field
                                M_zface(i, j, k, 0) /= M_magnitude_normalized;
                                M_zface(i, j, k, 1) /= M_magnitude_normalized;
                                M_zface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_zface
                        // x,y,z component on z-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_error_zface(i, j, k, icomp) = amrex::Math::abs((M_zface(i, j, k, icomp) - M_prev_zface(i, j, k, icomp))) / mag_Ms_zface_arr(i,j,k);
                        }
                    }
                });
        }

        // update H
        for (MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

            auto& mag_Ms_xface_mf = macroscopic_properties->getmag_Ms_mf(0);
            auto& mag_Ms_yface_mf = macroscopic_properties->getmag_Ms_mf(1);
            auto& mag_Ms_zface_mf = macroscopic_properties->getmag_Ms_mf(2);

            // extract material properties
            Array4<Real> const& mag_Ms_xface_arr = mag_Ms_xface_mf.array(mfi);
            Array4<Real> const& mag_Ms_yface_arr = mag_Ms_yface_mf.array(mfi);
            Array4<Real> const& mag_Ms_zface_arr = mag_Ms_zface_mf.array(mfi);

            // Extract field data for this grid/tile
            Array4<Real> const &Hx = Hfield[0]->array(mfi);
            Array4<Real> const &Hy = Hfield[1]->array(mfi);
            Array4<Real> const &Hz = Hfield[2]->array(mfi);
            Array4<Real> const &Hx_old = Hfield_old[0]->array(mfi);
            Array4<Real> const &Hy_old = Hfield_old[1]->array(mfi);
            Array4<Real> const &Hz_old = Hfield_old[2]->array(mfi);
            Array4<Real> const &Ex = Efield[0]->array(mfi);
            Array4<Real> const &Ey = Efield[1]->array(mfi);
            Array4<Real> const &Ez = Efield[2]->array(mfi);
            Array4<Real> const &M_xface = Mfield[0]->array(mfi);         // note M_xface include x,y,z components at |_x faces
            Array4<Real> const &M_yface = Mfield[1]->array(mfi);         // note M_yface include x,y,z components at |_y faces
            Array4<Real> const &M_zface = Mfield[2]->array(mfi);         // note M_zface include x,y,z components at |_z faces
            Array4<Real> const &M_xface_old = Mfield_old[0]->array(mfi); // note M_xface_old include x,y,z components at |_x faces
            Array4<Real> const &M_yface_old = Mfield_old[1]->array(mfi); // note M_yface_old include x,y,z components at |_y faces
            Array4<Real> const &M_zface_old = Mfield_old[2]->array(mfi); // note M_zface_old include x,y,z components at |_z faces

            // Extract stencil coefficients
            amrex::Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            amrex::Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            amrex::Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // Extract tileboxes for which to loop
            amrex::IntVect Hxnodal = Hfield[0]->ixType().toIntVect();
            amrex::IntVect Hynodal = Hfield[1]->ixType().toIntVect();
            amrex::IntVect Hznodal = Hfield[2]->ixType().toIntVect();
            Box const &tbx = mfi.tilebox(Hxnodal);
            Box const &tby = mfi.tilebox(Hynodal);
            Box const &tbz = mfi.tilebox(Hznodal);

            amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);

            amrex::Real const mu0_inv = 1. / PhysConst::mu0;

            // Loop over the cells and update the fields
            amrex::ParallelFor(tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    if (mag_Ms_xface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                        amrex::Real mu_arrx = CoarsenIO::Interp( mu_arr, mu_stag, Hx_stag, macro_cr, i, j, k, 0);
                        Hx(i, j, k) = Hx_old(i, j, k) + 1. / mu_arrx * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                                           - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                    } else if (mag_Ms_xface_arr(i,j,k) > 0){ // magnetic region
                        Hx(i, j, k) = Hx_old(i, j, k) + mu0_inv * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                                      - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                        if (coupling == 1) {
                            Hx(i, j, k) += - M_xface(i, j, k, 0) + M_xface_old(i, j, k, 0);
                        }
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    if (mag_Ms_yface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                        amrex::Real mu_arry = CoarsenIO::Interp( mu_arr, mu_stag, Hy_stag, macro_cr, i, j, k, 0);
                        Hy(i, j, k) = Hy_old(i, j, k) + 1. / mu_arry * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                                           - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                    } else if (mag_Ms_yface_arr(i,j,k) > 0){ // magnetic region
                        Hy(i, j, k) = Hy_old(i, j, k) + mu0_inv * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                                      - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                        if (coupling == 1){
                            Hy(i, j, k) += - M_yface(i, j, k, 1) + M_yface_old(i, j, k, 1);
                        }
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    if (mag_Ms_zface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                        amrex::Real mu_arrz = CoarsenIO::Interp( mu_arr, mu_stag, Hz_stag, macro_cr, i, j, k, 0);
                        Hz(i, j, k) = Hz_old(i, j, k) + 1. / mu_arrz * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                                           - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                    } else if (mag_Ms_zface_arr(i,j,k) > 0){ // magnetic region
                        Hz(i, j, k) = Hz_old(i, j, k) + mu0_inv * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                                      - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                        if (coupling == 1){
                            Hz(i, j, k) += - M_zface(i, j, k, 2) + M_zface_old(i, j, k, 2);
                        }
                    }
                }

            );
        }

        // Check the error between Mfield and Mfield_prev and decide whether another iteration is needed
        amrex::Real M_iter_maxerror = -1._rt;
        for (int iface = 0; iface < 3; iface++){
            for (int jcomp = 0; jcomp < 3; jcomp++){
                Real M_iter_error = Mfield_error[iface]->norm0(jcomp);
                if (M_iter_error >= M_iter_maxerror){
                    M_iter_maxerror = M_iter_error;
                }
            }
        }

        if (M_iter_maxerror <= M_tol){

            stop_iter = 1;

            // normalize M
            if (M_normalization == 2){

                for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

                    auto& mag_Ms_xface_mf = macroscopic_properties->getmag_Ms_mf(0);
                    auto& mag_Ms_yface_mf = macroscopic_properties->getmag_Ms_mf(1);
                    auto& mag_Ms_zface_mf = macroscopic_properties->getmag_Ms_mf(2);

                    // extract material properties
                    Array4<Real> const& mag_Ms_xface_arr = mag_Ms_xface_mf.array(mfi);
                    Array4<Real> const& mag_Ms_yface_arr = mag_Ms_yface_mf.array(mfi);
                    Array4<Real> const& mag_Ms_zface_arr = mag_Ms_zface_mf.array(mfi);

                    // extract field data
                    Array4<Real> const &M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
                    Array4<Real> const &M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
                    Array4<Real> const &M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces

                    // extract tileboxes for which to loop
                    amrex::IntVect Mxface_stag = Mfield[0]->ixType().toIntVect();
                    amrex::IntVect Myface_stag = Mfield[1]->ixType().toIntVect();
                    amrex::IntVect Mzface_stag = Mfield[2]->ixType().toIntVect();
                    Box const &tbx = mfi.tilebox(Mxface_stag); /* just define which grid type */
                    Box const &tby = mfi.tilebox(Myface_stag);
                    Box const &tbz = mfi.tilebox(Mzface_stag);

                    // loop over cells and update fields
                    amrex::ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (mag_Ms_xface_arr(i,j,k) > 0._rt){
                                // temporary normalized magnitude of M_xface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_xface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_xface_arr(i,j,k);

                                // check the normalized error
                                if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_xface field");
                                }
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (mag_Ms_yface_arr(i,j,k) > 0._rt){
                                // temporary normalized magnitude of M_yface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_yface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_yface_arr(i,j,k);

                                // check the normalized error
                                if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_yface field");
                                }
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (mag_Ms_zface_arr(i,j,k) > 0._rt){
                                // temporary normalized magnitude of M_zface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_zface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_zface_arr(i,j,k);

                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_zface field");
                                }
                                // normalize the M_zface field
                                M_zface(i, j, k, 0) /= M_magnitude_normalized;
                                M_zface(i, j, k, 1) /= M_magnitude_normalized;
                                M_zface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        });
                }
            }
        }
        else{
            const auto& period = warpx.Geom(lev).periodicity();
            // Copy Mfield to Mfield_previous and fill periodic/interior ghost cells
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(*Mfield_prev[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
                (*Mfield_prev[i]).FillBoundary(Mfield[i]->nGrowVect(), period);
            }
        }

        if (M_iter >= M_max_iter){
            amrex::Abort("The M_iter exceeds the M_max_iter");
            amrex::Print() << "The M_iter = " << M_iter << " exceeds the M_max_iter = " << M_max_iter << std::endl;
        }
        else{
            M_iter++;
            amrex::Print() << "Finish " << M_iter << " times iteration with M_iter_maxerror = " << M_iter_maxerror << " and M_tol = " << M_tol << std::endl;
        }

    } // end the iteration

    // update B
    for (MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

        auto& mag_Ms_xface_mf = macroscopic_properties->getmag_Ms_mf(0);
        auto& mag_Ms_yface_mf = macroscopic_properties->getmag_Ms_mf(1);
        auto& mag_Ms_zface_mf = macroscopic_properties->getmag_Ms_mf(2);

        // Extract field data for this grid/tile
        Array4<Real> const& mag_Ms_xface_arr = mag_Ms_xface_mf.array(mfi);
        Array4<Real> const& mag_Ms_yface_arr = mag_Ms_yface_mf.array(mfi);
        Array4<Real> const& mag_Ms_zface_arr = mag_Ms_zface_mf.array(mfi);
        Array4<Real> const &Hx = Hfield[0]->array(mfi);
        Array4<Real> const &Hy = Hfield[1]->array(mfi);
        Array4<Real> const &Hz = Hfield[2]->array(mfi);
        Array4<Real> const &Bx = Bfield[0]->array(mfi);
        Array4<Real> const &By = Bfield[1]->array(mfi);
        Array4<Real> const &Bz = Bfield[2]->array(mfi);
        Array4<Real> const &M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
        Array4<Real> const &M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
        Array4<Real> const &M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces

        // Extract tileboxes for which to loop
        amrex::IntVect Bxnodal = Bfield[0]->ixType().toIntVect();
        amrex::IntVect Bynodal = Bfield[1]->ixType().toIntVect();
        amrex::IntVect Bznodal = Bfield[2]->ixType().toIntVect();
        Box const &tbx = mfi.tilebox(Bxnodal);
        Box const &tby = mfi.tilebox(Bynodal);
        Box const &tbz = mfi.tilebox(Bznodal);

        amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                if (mag_Ms_xface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrx = CoarsenIO::Interp( mu_arr, mu_stag, Bx_stag, macro_cr, i, j, k, 0);
                    Bx(i, j, k) = mu_arrx * Hx(i, j, k);
                } else if (mag_Ms_xface_arr(i,j,k) > 0){
                    Bx(i, j, k) = PhysConst::mu0 * (M_xface(i, j, k, 0) + Hx(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                if (mag_Ms_yface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arry = CoarsenIO::Interp( mu_arr, mu_stag, By_stag, macro_cr, i, j, k, 0);
                    By(i, j, k) =  mu_arry * Hy(i, j, k);
                } else if (mag_Ms_yface_arr(i,j,k) > 0){
                    By(i, j, k) = PhysConst::mu0 * (M_yface(i, j, k, 1) + Hy(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                if (mag_Ms_zface_arr(i,j,k) == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrz = CoarsenIO::Interp( mu_arr, mu_stag, Bz_stag, macro_cr, i, j, k, 0);
                    Bz(i, j, k) = mu_arrz * Hz(i, j, k);
                } else if (mag_Ms_zface_arr(i,j,k) > 0){
                    Bz(i, j, k) = PhysConst::mu0 * (M_zface(i, j, k, 2) + Hz(i, j, k));
                }
            }

        );
    }
}
#endif // ifdef WARPX_MAG_LLG
#endif // ifndef WARPX_DIM_RZ
