/* copyright
blank
*/

#include "WarpX.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"

#include "Utils/WarpXConst.H"
#include "Utils/CoarsenIO.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update H and M fields with iterative correction, over one timestep
 */

#ifdef WARPX_MAG_LLG

void FiniteDifferenceSolver::MacroscopicEvolveHM_2nd(
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield, // Mfield contains three components MultiFab
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties) {

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee){
        MacroscopicEvolveHMCartesian_2nd<CartesianYeeAlgorithm>(Mfield, Hfield, Bfield, H_biasfield, Efield, dt, macroscopic_properties);
    } else {
        amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
} // closes function MacroscopicEvolveHM_2nd
#endif
#ifdef WARPX_MAG_LLG
template <typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveHMCartesian_2nd(
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

    amrex::GpuArray<int, 3> const& mag_Ms_stag    = macroscopic_properties->mag_Ms_IndexType;
    amrex::GpuArray<int, 3> const& mag_alpha_stag = macroscopic_properties->mag_alpha_IndexType;
    amrex::GpuArray<int, 3> const& mag_gamma_stag = macroscopic_properties->mag_gamma_IndexType;
    amrex::GpuArray<int, 3> const& mag_exchange_stag  = macroscopic_properties->mag_exchange_IndexType;
    amrex::GpuArray<int, 3> const& mag_anisotropy_stag   = macroscopic_properties->mag_anisotropy_IndexType;
    amrex::GpuArray<int, 3> const& mu_stag        = macroscopic_properties->mu_IndexType;
    amrex::GpuArray<int, 3> const& Mx_stag        = macroscopic_properties->Mx_IndexType;
    amrex::GpuArray<int, 3> const& My_stag        = macroscopic_properties->My_IndexType;
    amrex::GpuArray<int, 3> const& Mz_stag        = macroscopic_properties->Mz_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr       = macroscopic_properties->macro_cr_ratio;
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

    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(*a_temp_static[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
        auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
        auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
        auto& mag_exchange_mf = macroscopic_properties->getmag_exchange_mf();
        auto& mag_anisotropy_mf = macroscopic_properties->getmag_anisotropy_mf();

        // extract material properties
        Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
        Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
        Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);
        Array4<Real> const& mag_exchange_arr = mag_exchange_mf.array(mfi);
        Array4<Real> const& mag_anisotropy_arr = mag_anisotropy_mf.array(mfi);

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
        Box const &tbx = mfi.tilebox(Mfield[0]->ixType().toIntVect()); /* just define which grid type */
        Box const &tby = mfi.tilebox(Mfield[1]->ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Mfield[2]->ixType().toIntVect());

        // Extract stencil coefficients for calculating the exchange field H_exchange and the anisotropy field H_anisotropy
        amrex::Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        amrex::Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        amrex::Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();

        // loop over cells and update fields
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real mag_Ms_arrx    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arrx = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arrx = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arrx    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arrx    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mx_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrx > 0._rt){

                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy
                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_old);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange
                        if (mag_exchange_arrx == 0._rt) amrex::Abort("The mag_exchange_arrx is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrx / PhysConst::mu0 / mag_Ms_arrx / mag_Ms_arrx;
                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy
                        if (mag_anisotropy_arrx == 0._rt) amrex::Abort("The mag_anisotropy_arrx is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_xface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arrx / PhysConst::mu0 / mag_Ms_arrx / mag_Ms_arrx;
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arrx;
                    // a_temp_static_coeff does not change in the current step for SATURATED materials; but it does change for UNSATURATED ones
                    amrex::Real a_temp_static_coeff = mag_alpha_arrx / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrx) / 2._rt;

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

                amrex::Real mag_Ms_arry    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arry = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arry = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arry    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arry    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, My_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arry > 0._rt){

                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_old);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange
                        if (mag_exchange_arry == 0._rt) amrex::Abort("The mag_exchange_arry is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arry / PhysConst::mu0 / mag_Ms_arry / mag_Ms_arry;
                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy
                        if (mag_anisotropy_arry == 0._rt) amrex::Abort("The mag_anisotropy_arry is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_yface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arry / PhysConst::mu0 / mag_Ms_arry / mag_Ms_arry;
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    // note the unsaturated case is less usefull in real devices
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arry;
                    amrex::Real a_temp_static_coeff = mag_alpha_arry / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_arry) / 2._rt;

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

                amrex::Real mag_Ms_arrz    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arrz = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arrz = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arrz    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arrz    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mz_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrz > 0._rt){

                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                    if (coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_old);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_old);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_old);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange
                        if (mag_exchange_arrz == 0._rt) amrex::Abort("The mag_exchange_arrz is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrz / PhysConst::mu0 / mag_Ms_arrz / mag_Ms_arrz;
                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy
                        if (mag_anisotropy_arrz == 0._rt) amrex::Abort("The mag_anisotropy_arrz is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_zface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arrz / PhysConst::mu0 / mag_Ms_arrz / mag_Ms_arrz;
                        Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arrz;
                    amrex::Real a_temp_static_coeff = mag_alpha_arrz / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrz) / 2._rt;

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

        warpx.FillBoundaryH(warpx.getngE());

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){
            auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
            auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
            auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
            auto& mag_exchange_mf = macroscopic_properties->getmag_exchange_mf();
            auto& mag_anisotropy_mf = macroscopic_properties->getmag_anisotropy_mf();

            // extract material properties
            Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
            Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
            Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);
            Array4<Real> const& mag_exchange_arr = mag_exchange_mf.array(mfi);
            Array4<Real> const& mag_anisotropy_arr = mag_anisotropy_mf.array(mfi);

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
            Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

            // Extract stencil coefficients for calculating the exchange field H_exchange and the anisotropy field H_anisotropy
            amrex::Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            amrex::Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            amrex::Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();

            // loop over cells and update fields
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    amrex::Real mag_Ms_arrx    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mx_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_alpha_arrx = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mx_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_gamma_arrx = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mx_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_exchange_arrx    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mx_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_anisotropy_arrx    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mx_stag, macro_cr, i, j, k, 0);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arrx > 0._rt){
                        // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz);
                        }

                        if (mag_exchange_coupling == 1){
                            // H_exchange
                            if (mag_exchange_arrx == 0._rt) amrex::Abort("The mag_exchange_arrx is 0.0 while including the exchange coupling term H_exchange for H_eff");
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrx / PhysConst::mu0 / mag_Ms_arrx / mag_Ms_arrx;
                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_xface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                        }

                        if (mag_anisotropy_coupling == 1){
                            // H_anisotropy
                            if (mag_anisotropy_arrx == 0._rt) amrex::Abort("The mag_anisotropy_arrx is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_xface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arrx / PhysConst::mu0 / mag_Ms_arrx / mag_Ms_arrx;
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrx) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_xface
                            // all components on x-faces of grid
                            a_temp_xface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_xface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_xface(i, j, k, comp)
                                                                                     + 0.5 * mag_alpha_arrx * 1. / std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) * M_old_xface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_xface from a and b using the updateM_field
                            // all components on x-faces of grid
                            M_xface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_xface, b_temp_static_xface);
                        }

                        // temporary normalized magnitude of M_xface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / mag_Ms_arrx;
                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
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
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrx);
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
                            M_error_xface(i, j, k, icomp) = amrex::Math::abs((M_xface(i, j, k, icomp) - M_prev_xface(i, j, k, icomp))) / mag_Ms_arrx;
                        }
                    }
                });

            amrex::ParallelFor(tby,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    amrex::Real mag_Ms_arry    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, My_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_alpha_arry = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, My_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_gamma_arry = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, My_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_exchange_arry    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, My_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_anisotropy_arry    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, My_stag, macro_cr, i, j, k, 0);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arry > 0._rt){
                        // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz);
                        }

                        if (mag_exchange_coupling == 1){
                            // H_exchange
                            if (mag_exchange_arry == 0._rt) amrex::Abort("The mag_exchange_arry is 0.0 while including the exchange coupling term H_exchange for H_eff");
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arry / PhysConst::mu0 / mag_Ms_arry / mag_Ms_arry;
                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_yface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                        }

                        if (mag_anisotropy_coupling == 1){
                            // H_anisotropy
                            if (mag_anisotropy_arry == 0._rt) amrex::Abort("The mag_anisotropy_arry is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_yface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arry / PhysConst::mu0 / mag_Ms_arry / mag_Ms_arry;
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arry) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_yface
                            // all components on y-faces of grid
                            a_temp_yface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_yface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_yface(i, j, k, comp)
                                                                                     + 0.5 * mag_alpha_arry * 1. / std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) * M_old_yface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_yface from a and b using the updateM_field
                            // all components on y-faces of grid
                            M_yface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_yface, b_temp_static_yface);
                        }

                        // temporary normalized magnitude of M_yface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / mag_Ms_arry;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
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
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arry);
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
                            M_error_yface(i, j, k, icomp) = amrex::Math::abs((M_yface(i, j, k, icomp) - M_prev_yface(i, j, k, icomp))) / mag_Ms_arry;
                        }
                    }
                });

            amrex::ParallelFor(tbz,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    amrex::Real mag_Ms_arrz    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mz_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_alpha_arrz = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mz_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_gamma_arrz = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mz_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_exchange_arrz    = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mz_stag, macro_cr, i, j, k, 0);
                    amrex::Real mag_anisotropy_arrz    = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mz_stag, macro_cr, i, j, k, 0);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arrz > 0._rt){
                        // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                        amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                        amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                        if (coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx);
                            Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy);
                            Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz);
                        }

                        if (mag_exchange_coupling == 1){
                            // H_exchange
                            if (mag_exchange_arrz == 0._rt) amrex::Abort("The mag_exchange_arrz is 0.0 while including the exchange coupling term H_exchange for H_eff");
                            amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrz / PhysConst::mu0 / mag_Ms_arrz / mag_Ms_arrz;
                            Hx_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 0);
                            Hy_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 1);
                            Hz_eff += H_exchange_coeff * T_Algo::Laplacian(M_zface, coefs_x, coefs_y, coefs_z, i, j, k, 2);
                        }

                        if (mag_anisotropy_coupling == 1){
                            // H_anisotropy
                            if (mag_anisotropy_arrz == 0._rt) amrex::Abort("The mag_anisotropy_arrz is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_zface(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * mag_anisotropy_arrz / PhysConst::mu0 / mag_Ms_arrz / mag_Ms_arrz;
                            Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrz) / 2._rt;

                        amrex::GpuArray<amrex::Real,3> H_eff;
                        H_eff[0] = Hx_eff;
                        H_eff[1] = Hy_eff;
                        H_eff[2] = Hz_eff;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_zface
                            // all components on z-faces of grid
                            a_temp_zface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff[comp] + a_temp_static_zface(i, j, k, comp))
                                                                              : -(dt * a_temp_dynamic_coeff * H_eff[comp] + 0.5 * a_temp_static_zface(i, j, k, comp)
                                                                                  + 0.5 * mag_alpha_arrz * 1. / std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) * M_old_zface(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_zface from a and b using the updateM_field
                            // all components on z-faces of grid
                            M_zface(i, j, k, comp) = MacroscopicProperties::updateM_field(i, j, k, comp, a_temp_zface, b_temp_static_zface);
                        }

                        // temporary normalized magnitude of M_zface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / mag_Ms_arrz;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error){
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
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
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrz);
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
                            M_error_zface(i, j, k, icomp) = amrex::Math::abs((M_zface(i, j, k, icomp) - M_prev_zface(i, j, k, icomp))) / mag_Ms_arrz;
                        }
                    }
                });
        }

        // update H
        for (MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){
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
            Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect());
            Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

            // read in Ms to decide if the grid is magnetic or not
            auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
            Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);

            // mu_mf will be imported but will only be called at grids where Ms == 0
            auto& mu_mf = macroscopic_properties->getmu_mf();
            Array4<Real> const& mu_arr = mu_mf.array(mfi);

            amrex::Real const mu0_inv = 1. / PhysConst::mu0;

            // Loop over the cells and update the fields
            amrex::ParallelFor(tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real mag_Ms_arrx    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mx_stag, macro_cr, i, j, k, 0);
                    if (mag_Ms_arrx == 0._rt){ // nonmagnetic region
                        Real mu_arrx    = CoarsenIO::Interp( mu_arr, mu_stag, Mx_stag, macro_cr, i, j, k, 0);
                        Hx(i, j, k) = Hx_old(i, j, k) + 1. / mu_arrx * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                                           - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                    } else if (mag_Ms_arrx > 0){ // magnetic region
                        Hx(i, j, k) = Hx_old(i, j, k) + mu0_inv * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                                      - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                        if (coupling == 1) {
                        Hx(i, j, k) += - M_xface(i, j, k, 0) + M_xface_old(i, j, k, 0);
                        }
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real mag_Ms_arry    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, My_stag, macro_cr, i, j, k, 0);
                    if (mag_Ms_arry == 0._rt){ // nonmagnetic region
                        Real mu_arry    = CoarsenIO::Interp( mu_arr, mu_stag, My_stag, macro_cr, i, j, k, 0);
                        Hy(i, j, k) = Hy_old(i, j, k) + 1. / mu_arry * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                                           - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                    } else if (mag_Ms_arry > 0){ // magnetic region
                        Hy(i, j, k) = Hy_old(i, j, k) + mu0_inv * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                                      - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                        if (coupling == 1){
                            Hy(i, j, k) += - M_yface(i, j, k, 1) + M_yface_old(i, j, k, 1);
                        }
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real mag_Ms_arrz    = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mz_stag, macro_cr, i, j, k, 0);
                    if (mag_Ms_arrz == 0._rt){ // nonmagnetic region
                        Real mu_arrz    = CoarsenIO::Interp( mu_arr, mu_stag, Mz_stag, macro_cr, i, j, k, 0);
                        Hz(i, j, k) = Hz_old(i, j, k) + 1. / mu_arrz * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                                           - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                    } else if (mag_Ms_arrz > 0){ // magnetic region
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
                    auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
                    // extract material properties
                    Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);

                    // extract field data
                    Array4<Real> const &M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
                    Array4<Real> const &M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
                    Array4<Real> const &M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces

                    // extract tileboxes for which to loop
                    Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect()); /* just define which grid type */
                    Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
                    Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

                    // loop over cells and update fields
                    amrex::ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

                            if (mag_Ms_arrx > 0._rt){
                                // temporary normalized magnitude of M_xface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_xface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arrx;

                                // check the normalized error
                                if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                    printf("i = %d, j=%d, k=%d\n", i, j, k);
                                    printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                                    amrex::Abort("Exceed the normalized error of the M_xface field");
                                }
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);

                            if (mag_Ms_arry > 0._rt){
                                // temporary normalized magnitude of M_yface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_yface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arry;

                                // check the normalized error
                                if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error){
                                    printf("i = %d, j=%d, k=%d\n", i, j, k);
                                    printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                                    amrex::Abort("Exceed the normalized error of the M_yface field");
                                }
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);

                            if (mag_Ms_arrz > 0._rt){
                                // temporary normalized magnitude of M_zface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_zface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arrz;

                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error){
                                    printf("i = %d, j=%d, k=%d\n", i, j, k);
                                    printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
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
            // Copy Mfield to Mfield_previous
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(*Mfield_prev[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
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
        // Extract field data for this grid/tile
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
        Box const &tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const &tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // read in Ms to decide if the grid is magnetic or not
        auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
        Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);

        // mu_mf will be imported but will only be called at grids where Ms == 0
        auto& mu_mf = macroscopic_properties->getmu_mf();
        Array4<Real> const& mu_arr = mu_mf.array(mfi);

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real mag_Ms_arrx = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1,0,0), mag_Ms_arr);
                if (mag_Ms_arrx == 0._rt){ // nonmagnetic region
                    Real mu_arrx = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1,0,0), mu_arr);
                    Bx(i, j, k) = mu_arrx * Hx(i, j, k);
                } else if (mag_Ms_arrx > 0){
                    Bx(i, j, k) = PhysConst::mu0 * (M_xface(i, j, k, 0) + Hx(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real mag_Ms_arry = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,1,0), mag_Ms_arr);
                if (mag_Ms_arry == 0._rt){ // nonmagnetic region
                    Real mu_arry = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,1,0), mu_arr);
                    By(i, j, k) =  mu_arry * Hy(i, j, k);
                } else if (mag_Ms_arry > 0){
                    By(i, j, k) = PhysConst::mu0 * (M_yface(i, j, k, 1) + Hy(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real mag_Ms_arrz = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,0,1), mag_Ms_arr);
                if (mag_Ms_arrz == 0._rt){ // nonmagnetic region
                    Real mu_arrz = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,0,1), mu_arr);
                    Bz(i, j, k) = mu_arrz * Hz(i, j, k);
                } else if (mag_Ms_arrz > 0){
                    Bz(i, j, k) = PhysConst::mu0 * (M_zface(i, j, k, 2) + Hz(i, j, k));
                }
            }

        );
    }
}
#endif
