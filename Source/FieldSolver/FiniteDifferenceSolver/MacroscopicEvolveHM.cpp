/*
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#endif
#include "Utils/WarpXConst.H"
#include "Utils/CoarsenIO.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update H and M fields without iterative correction, over one timestep
 */

#ifdef WARPX_MAG_LLG

void FiniteDifferenceSolver::MacroscopicEvolveHM(
    // The MField here is a vector of three multifabs, with M on each face.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield, // H Maxwell
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties)
{

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveHMCartesian<CartesianYeeAlgorithm>(Mfield, Hfield, Bfield, H_biasfield, Efield, dt, macroscopic_properties);
    }
    else
    {
        amrex::Abort("Only yee algorithm is compatible for H and M updates.");
    }
} // closes function EvolveM
#endif

#ifdef WARPX_MAG_LLG
template <typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveHMCartesian(
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield, // H Maxwell
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties)
{

    auto &warpx = WarpX::GetInstance();
    int coupling = warpx.mag_LLG_coupling;
    int M_normalization = warpx.mag_M_normalization;
    int mag_exchange_coupling = warpx.mag_LLG_exchange_coupling;
    int mag_anisotropy_coupling = warpx.mag_LLG_anisotropy_coupling;

    // temporary Multifab storing M from previous timestep (old_time) before updating to M(new_time)
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_old; // Mfield_old is M(old_time)

    amrex::GpuArray<int, 3> const& mag_Ms_stag         = macroscopic_properties->mag_Ms_IndexType;
    amrex::GpuArray<int, 3> const& mag_alpha_stag      = macroscopic_properties->mag_alpha_IndexType;
    amrex::GpuArray<int, 3> const& mag_gamma_stag      = macroscopic_properties->mag_gamma_IndexType;
    amrex::GpuArray<int, 3> const& mag_exchange_stag   = macroscopic_properties->mag_exchange_IndexType;
    amrex::GpuArray<int, 3> const& mag_anisotropy_stag = macroscopic_properties->mag_anisotropy_IndexType;
    amrex::GpuArray<int, 3> const& Mx_stag             = macroscopic_properties->Mx_IndexType;
    amrex::GpuArray<int, 3> const& My_stag             = macroscopic_properties->My_IndexType;
    amrex::GpuArray<int, 3> const& Mz_stag             = macroscopic_properties->Mz_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr            = macroscopic_properties->macro_cr_ratio;
    amrex::GpuArray<amrex::Real, 3> const& anisotropy_axis = macroscopic_properties->mag_LLG_anisotropy_axis;

    for (int i = 0; i < 3; i++)
    {
        // Mfield_old is M(n)
        Mfield_old[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        // initialize temporary multifab, Mfield_old, with values from Mfield(old_time)
        MultiFab::Copy(*Mfield_old[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
    }

    // obtain the maximum relative amount we let M deviate from Ms before aborting
    amrex::Real mag_normalized_error = macroscopic_properties->getmag_normalized_error();

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

    for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
    {
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
        Array4<Real> const &Hx = Hfield[0]->array(mfi);
        Array4<Real> const &Hy = Hfield[1]->array(mfi);
        Array4<Real> const &Hz = Hfield[2]->array(mfi);
        Array4<Real> const &M_xface = Mfield[0]->array(mfi);         // note M_xface include x,y,z components at |_x faces
        Array4<Real> const &M_yface = Mfield[1]->array(mfi);         // note M_yface include x,y,z components at |_y faces
        Array4<Real> const &M_zface = Mfield[2]->array(mfi);         // note M_zface include x,y,z components at |_z faces
        Array4<Real> const &M_xface_old = Mfield_old[0]->array(mfi); // note M_xface_old include x,y,z components at |_x faces
        Array4<Real> const &M_yface_old = Mfield_old[1]->array(mfi); // note M_yface_old include x,y,z components at |_y faces
        Array4<Real> const &M_zface_old = Mfield_old[2]->array(mfi); // note M_zface_old include x,y,z components at |_z faces
        Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi);    // Hx_bias is the x component at |_x faces
        Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi);    // Hy_bias is the y component at |_y faces
        Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi);    // Hz_bias is the z component at |_z faces

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

                amrex::Real mag_Ms_arrx         = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arrx      = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arrx      = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arrx   = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mx_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arrx = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mx_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrx > 0._rt)
                {
                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

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

                    // magnetic material properties mag_alpha and mag_Ms are defined at faces
                    // removed the interpolation from version with cell-nodal material properties
                    amrex::Real mag_gammaL = mag_gamma_arrx / (1._rt + std::pow(mag_alpha_arrx, 2._rt));

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arrx;
                    amrex::Real Gil_damp = PhysConst::mu0 * mag_gammaL * mag_alpha_arrx / M_magnitude;

                    // now you have access to use M_xface(i,j,k,0) M_xface(i,j,k,1), M_xface(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below
                    // x component on x-faces of grid
                    M_xface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff)
                                         - M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff));

                    // y component on x-faces of grid
                    M_xface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff)
                                         - M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff));

                    // z component on x-faces of grid
                    M_xface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff)
                                         - M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_xface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / mag_Ms_arrx;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_xface field");
                        }
                        // normalize the M_xface field
                        M_xface(i, j, k, 0) /= M_magnitude_normalized;
                        M_xface(i, j, k, 1) /= M_magnitude_normalized;
                        M_xface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > (1._rt + mag_normalized_error))
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrx);
                            amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + mag_normalized_error) )
                        {
                            // normalize the M_xface field
                            M_xface(i, j, k, 0) /= M_magnitude_normalized;
                            M_xface(i, j, k, 1) /= M_magnitude_normalized;
                            M_xface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                } // end if (mag_Ms_arrx(i,j,k) > 0...
            });

        amrex::ParallelFor(tby,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real mag_Ms_arry         = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arry      = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arry      = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arry   = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, My_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arry = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, My_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arry > 0._rt)
                {
                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

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

                    // magnetic material properties mag_alpha and mag_Ms are defined at faces
                    // removed the interpolation from version with cell-nodal material properties
                    amrex::Real mag_gammaL = mag_gamma_arry / (1._rt + std::pow(mag_alpha_arry, 2._rt));

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arry;
                    amrex::Real Gil_damp = PhysConst::mu0 * mag_gammaL * mag_alpha_arry / M_magnitude;

                    // x component on y-faces of grid
                    M_yface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff)
                                         - M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff));

                    // y component on y-faces of grid
                    M_yface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff)
                                         - M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff));

                    // z component on y-faces of grid
                    M_yface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff)
                                         - M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_yface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / mag_Ms_arry;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_yface field");
                        }
                        // normalize the M_yface field
                        M_yface(i, j, k, 0) /= M_magnitude_normalized;
                        M_yface(i, j, k, 1) /= M_magnitude_normalized;
                        M_yface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > 1._rt + mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arry);
                            amrex::Abort("Caution: Unsaturated material has M_yface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error)
                        {
                            // normalize the M_yface field
                            M_yface(i, j, k, 0) /= M_magnitude_normalized;
                            M_yface(i, j, k, 1) /= M_magnitude_normalized;
                            M_yface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                } // end if (mag_Ms_arry(i,j,k) > 0...
            });

        amrex::ParallelFor(tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real mag_Ms_arrz         = CoarsenIO::Interp( mag_Ms_arr, mag_Ms_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_alpha_arrz      = CoarsenIO::Interp( mag_alpha_arr, mag_alpha_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_gamma_arrz      = CoarsenIO::Interp( mag_gamma_arr, mag_gamma_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_exchange_arrz   = CoarsenIO::Interp( mag_exchange_arr, mag_exchange_stag, Mz_stag, macro_cr, i, j, k, 0);
                amrex::Real mag_anisotropy_arrz = CoarsenIO::Interp( mag_anisotropy_arr, mag_anisotropy_stag, Mz_stag, macro_cr, i, j, k, 0);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrz > 0._rt)
                {
                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

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

                    // magnetic material properties mag_alpha and mag_Ms are defined at faces
                    // removed the interpolation from version with cell-nodal material properties
                    amrex::Real mag_gammaL = mag_gamma_arrz / (1._rt + std::pow(mag_alpha_arrz, 2._rt));

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt))
                                                              : mag_Ms_arrz;
                    amrex::Real Gil_damp = PhysConst::mu0 * mag_gammaL * mag_alpha_arrz / M_magnitude;

                    // x component on z-faces of grid
                    M_zface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff)
                                         - M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff));

                    // y component on z-faces of grid
                    M_zface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff)
                                         - M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff));

                    // z component on z-faces of grid
                    M_zface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff)
                                         - M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_zface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / mag_Ms_arrz;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_zface field");
                        }
                        // normalize the M_zface field
                        M_zface(i, j, k, 0) /= M_magnitude_normalized;
                        M_zface(i, j, k, 1) /= M_magnitude_normalized;
                        M_zface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > 1._rt + mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrz);
                            amrex::Abort("Caution: Unsaturated material has M_zface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error)
                        {
                            // normalize the M_zface field
                            M_zface(i, j, k, 0) /= M_magnitude_normalized;
                            M_zface(i, j, k, 1) /= M_magnitude_normalized;
                            M_zface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                } // end if (mag_Ms_arrz(i,j,k) > 0...
            });
    }
    // Update H(new_time) = f(H(old_time), M(new_time), M(old_time), E(old_time))
    for (MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Extract field data for this grid/tile
        Array4<Real> const &Hx = Hfield[0]->array(mfi);
        Array4<Real> const &Hy = Hfield[1]->array(mfi);
        Array4<Real> const &Hz = Hfield[2]->array(mfi);
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
                Real mag_Ms_arrx = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1,0,0), mag_Ms_arr);
                if (mag_Ms_arrx == 0._rt){ // nonmagnetic region
                    Real mu_arrx = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1,0,0), mu_arr);
                    Hx(i, j, k) += 1. / mu_arrx * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                      - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                } else if (mag_Ms_arrx > 0){ // magnetic region
                    Hx(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                 - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                    if (coupling == 1) {
                        Hx(i, j, k) += - M_xface(i, j, k, 0) + M_xface_old(i, j, k, 0);
                    }
                }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real mag_Ms_arry = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,1,0), mag_Ms_arr);
                if (mag_Ms_arry == 0._rt){ // nonmagnetic region
                    Real mu_arry = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,1,0), mu_arr);
                    Hy(i, j, k) += 1. / mu_arry * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                      - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                } else if (mag_Ms_arry > 0){ // magnetic region
                    Hy(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                 - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                    if (coupling == 1){
                        Hy(i, j, k) += - M_yface(i, j, k, 1) + M_yface_old(i, j, k, 1);
                    }
                }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real mag_Ms_arrz = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,0,1), mag_Ms_arr);
                if (mag_Ms_arrz == 0._rt){ // nonmagnetic region
                    Real mu_arrz = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0,0,1), mu_arr);
                    Hz(i, j, k) += 1. / mu_arrz * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                      - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                } else if (mag_Ms_arrz > 0){ // magnetic region
                    Hz(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                 - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                    if (coupling == 1){
                        Hz(i, j, k) += - M_zface(i, j, k, 2) + M_zface_old(i, j, k, 2);
                    }
                }
            });
    }

    // update B
    for (MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
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
            });
    }
}
#endif
