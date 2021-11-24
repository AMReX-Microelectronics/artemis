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
#include "Utils/WarpXUtil.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update H and M fields without iterative correction, over one timestep
 */

#ifndef WARPX_DIM_RZ
#ifdef WARPX_MAG_LLG

void FiniteDifferenceSolver::MacroscopicEvolveHM(
    // The MField here is a vector of three multifabs, with M on each face.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    int lev,
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
        MacroscopicEvolveHMCartesian<CartesianYeeAlgorithm>(lev, Mfield, Hfield, Bfield, H_biasfield, Efield, dt, macroscopic_properties);
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
    int lev,
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

    amrex::GpuArray<int, 3> const& mag_alpha_stag      = macroscopic_properties->mag_alpha_IndexType;
    amrex::GpuArray<int, 3> const& mag_gamma_stag      = macroscopic_properties->mag_gamma_IndexType;
    amrex::GpuArray<int, 3> const& mag_exchange_stag   = macroscopic_properties->mag_exchange_IndexType;
    amrex::GpuArray<int, 3> const& mag_anisotropy_stag = macroscopic_properties->mag_anisotropy_IndexType;
    amrex::GpuArray<int, 3> const& mu_stag             = macroscopic_properties->mu_IndexType;
    amrex::GpuArray<int, 3> const& Mx_stag             = macroscopic_properties->Mx_IndexType;
    amrex::GpuArray<int, 3> const& My_stag             = macroscopic_properties->My_IndexType;
    amrex::GpuArray<int, 3> const& Mz_stag             = macroscopic_properties->Mz_IndexType;
    amrex::GpuArray<int, 3> const& Hx_stag             = macroscopic_properties->Hx_IndexType;
    amrex::GpuArray<int, 3> const& Hy_stag             = macroscopic_properties->Hy_IndexType;
    amrex::GpuArray<int, 3> const& Hz_stag             = macroscopic_properties->Hz_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag             = macroscopic_properties->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag             = macroscopic_properties->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag             = macroscopic_properties->Bz_IndexType;
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

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

    for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
    {
        auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
        auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
        auto& mag_exchange_mf = macroscopic_properties->getmag_exchange_mf();
        auto& mag_anisotropy_mf = macroscopic_properties->getmag_anisotropy_mf();

        // extract material properties
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
        Array4<Real> const &M_old_xface = Mfield_old[0]->array(mfi); // note M_old_xface include x,y,z components at |_x faces
        Array4<Real> const &M_old_yface = Mfield_old[1]->array(mfi); // note M_old_yface include x,y,z components at |_y faces
        Array4<Real> const &M_old_zface = Mfield_old[2]->array(mfi); // note M_old_zface include x,y,z components at |_z faces
        Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi);    // Hx_bias is the x component at |_x faces
        Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi);    // Hy_bias is the y component at |_y faces
        Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi);    // Hz_bias is the z component at |_z faces

        amrex::IntVect Mxface_stag = Mfield[0]->ixType().toIntVect();
        amrex::IntVect Myface_stag = Mfield[1]->ixType().toIntVect();
        amrex::IntVect Mzface_stag = Mfield[2]->ixType().toIntVect();

        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect()); /* just define which grid type */
        Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

        // Extract stencil coefficients for calculating the exchange field H_exchange and the anisotropy field H_anisotropy
        amrex::Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        amrex::Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        amrex::Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto mag_parser = macroscopic_properties->m_mag_Ms_parser->compile<3>();

        // loop over cells and update fields
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Mx_stag, problo, dx, x, y, z);
                amrex::Real mag_Ms_arrx = mag_parser(x,y,z);
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
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange - use M^(old_time)
                        if (mag_exchange_arrx == 0._rt) amrex::Abort("The mag_exchange_arrx is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrx / PhysConst::mu0 / mag_Ms_arrx / mag_Ms_arrx;

                        WarpXUtilAlgo::getCellCoordinates(i-1, j, k, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i+1, j, k, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j-1, k, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j+1, k, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k-1, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_z = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k+1, Mx_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_z = mag_parser(x,y,z);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 0); //Last argument is nodality -- xface = 0
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 0); //Last argument is nodality -- xface = 0
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_xface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 0); //Last argument is nodality -- xface = 0
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy - use M^(old_time)
                        if (mag_anisotropy_arrx == 0._rt) amrex::Abort("The mag_anisotropy_arrx is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_old_xface(i, j, k, comp) * anisotropy_axis[comp];
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

                    // now you have access to use M_old_xface(i,j,k,:), Hx_eff, Hy_eff, and Hz_eff on the RHS of these update lines below
                    // x component on x-faces of grid
                    M_xface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_xface(i, j, k, 1) * Hz_eff - M_old_xface(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_old_xface(i, j, k, 1) * (M_old_xface(i, j, k, 0) * Hy_eff - M_old_xface(i, j, k, 1) * Hx_eff)
                                         - M_old_xface(i, j, k, 2) * (M_old_xface(i, j, k, 2) * Hx_eff - M_old_xface(i, j, k, 0) * Hz_eff));

                    // y component on x-faces of grid
                    M_xface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_xface(i, j, k, 2) * Hx_eff - M_old_xface(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_old_xface(i, j, k, 2) * (M_old_xface(i, j, k, 1) * Hz_eff - M_old_xface(i, j, k, 2) * Hy_eff)
                                         - M_old_xface(i, j, k, 0) * (M_old_xface(i, j, k, 0) * Hy_eff - M_old_xface(i, j, k, 1) * Hx_eff));

                    // z component on x-faces of grid
                    M_xface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_xface(i, j, k, 0) * Hy_eff - M_old_xface(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_old_xface(i, j, k, 0) * (M_old_xface(i, j, k, 2) * Hx_eff - M_old_xface(i, j, k, 0) * Hz_eff)
                                         - M_old_xface(i, j, k, 1) * (M_old_xface(i, j, k, 1) * Hz_eff - M_old_xface(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_xface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / mag_Ms_arrx;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
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

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, My_stag, problo, dx, x, y, z);
                amrex::Real mag_Ms_arry = mag_parser(x,y,z);
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
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange - use M^(old_time)
                        if (mag_exchange_arry == 0._rt) amrex::Abort("The mag_exchange_arry is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arry / PhysConst::mu0 / mag_Ms_arry / mag_Ms_arry;

                        WarpXUtilAlgo::getCellCoordinates(i-1, j, k, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i+1, j, k, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j-1, k, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j+1, k, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k-1, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_z = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k+1, My_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_z = mag_parser(x,y,z);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 1); //Last argument is nodality -- yface = 1
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 1); //Last argument is nodality -- yface = 1
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_yface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 1); //Last argument is nodality -- yface = 1
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy - use M^(old_time)
                        if (mag_anisotropy_arry == 0._rt) amrex::Abort("The mag_anisotropy_arry is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_old_yface(i, j, k, comp) * anisotropy_axis[comp];
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

                    // now you have access to use M_old_yface(i,j,k,:), Hx_eff, Hy_eff, and Hz_eff on the RHS of these update lines below
                    // x component on y-faces of grid
                    M_yface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_yface(i, j, k, 1) * Hz_eff - M_old_yface(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_old_yface(i, j, k, 1) * (M_old_yface(i, j, k, 0) * Hy_eff - M_old_yface(i, j, k, 1) * Hx_eff)
                                         - M_old_yface(i, j, k, 2) * (M_old_yface(i, j, k, 2) * Hx_eff - M_old_yface(i, j, k, 0) * Hz_eff));

                    // y component on y-faces of grid
                    M_yface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_yface(i, j, k, 2) * Hx_eff - M_old_yface(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_old_yface(i, j, k, 2) * (M_old_yface(i, j, k, 1) * Hz_eff - M_old_yface(i, j, k, 2) * Hy_eff)
                                         - M_old_yface(i, j, k, 0) * (M_old_yface(i, j, k, 0) * Hy_eff - M_old_yface(i, j, k, 1) * Hx_eff));

                    // z component on y-faces of grid
                    M_yface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_yface(i, j, k, 0) * Hy_eff - M_old_yface(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_old_yface(i, j, k, 0) * (M_old_yface(i, j, k, 2) * Hx_eff - M_old_yface(i, j, k, 0) * Hz_eff)
                                         - M_old_yface(i, j, k, 1) * (M_old_yface(i, j, k, 1) * Hz_eff - M_old_yface(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_yface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / mag_Ms_arry;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
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

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Mz_stag, problo, dx, x, y, z);
                amrex::Real mag_Ms_arrz = mag_parser(x,y,z);
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
                    amrex::Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx_bias);
                    amrex::Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy_bias);
                    amrex::Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz_bias);

                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell - use H^(old_time)
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz);
                    }

                    if (mag_exchange_coupling == 1){
                        // H_exchange - use M^(old_time)
                        if (mag_exchange_arrz == 0._rt) amrex::Abort("The mag_exchange_arrz is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        amrex::Real const H_exchange_coeff = 2.0 * mag_exchange_arrz / PhysConst::mu0 / mag_Ms_arrz / mag_Ms_arrz;

                        WarpXUtilAlgo::getCellCoordinates(i-1, j, k, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i+1, j, k, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_x = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j-1, k, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j+1, k, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_y = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k-1, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_lo_z = mag_parser(x,y,z);
                        WarpXUtilAlgo::getCellCoordinates(i, j, k+1, Mz_stag, problo, dx, x, y, z);
                        amrex::Real Ms_hi_z = mag_parser(x,y,z);

                        Hx_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 0, 2); //Last argument is nodality -- zface = 2
                        Hy_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 1, 2); //Last argument is nodality -- zface = 2
                        Hz_eff += H_exchange_coeff * T_Algo::Laplacian_Mag(M_old_zface, coefs_x, coefs_y, coefs_z, n_coefs_x, n_coefs_y, n_coefs_z, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, 2, 2); //Last argument is nodality -- zface = 2
                    }

                    if (mag_anisotropy_coupling == 1){
                        // H_anisotropy - use M^(old_time)
                        if (mag_anisotropy_arrz == 0._rt) amrex::Abort("The mag_anisotropy_arrz is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_old_zface(i, j, k, comp) * anisotropy_axis[comp];
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

                    // now you have access to use M_old_zface(i,j,k,:), Hx_eff, Hy_eff, and Hz_eff on the RHS of these update lines below
                    // x component on z-faces of grid
                    M_zface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_zface(i, j, k, 1) * Hz_eff - M_old_zface(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_old_zface(i, j, k, 1) * (M_old_zface(i, j, k, 0) * Hy_eff - M_old_zface(i, j, k, 1) * Hx_eff)
                                         - M_old_zface(i, j, k, 2) * (M_old_zface(i, j, k, 2) * Hx_eff - M_old_zface(i, j, k, 0) * Hz_eff));

                    // y component on z-faces of grid
                    M_zface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_zface(i, j, k, 2) * Hx_eff - M_old_zface(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_old_zface(i, j, k, 2) * (M_old_zface(i, j, k, 1) * Hz_eff - M_old_zface(i, j, k, 2) * Hy_eff)
                                         - M_old_zface(i, j, k, 0) * (M_old_zface(i, j, k, 0) * Hy_eff - M_old_zface(i, j, k, 1) * Hx_eff));

                    // z component on z-faces of grid
                    M_zface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gammaL) * (M_old_zface(i, j, k, 0) * Hy_eff - M_old_zface(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_old_zface(i, j, k, 0) * (M_old_zface(i, j, k, 2) * Hx_eff - M_old_zface(i, j, k, 0) * Hz_eff)
                                         - M_old_zface(i, j, k, 1) * (M_old_zface(i, j, k, 1) * Hz_eff - M_old_zface(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_zface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / mag_Ms_arrz;

                    if (M_normalization > 0)
                    {
                        // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
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

    amrex::MultiFab& mu_mf = macroscopic_properties->getmu_mf();
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
        Array4<Real> const &M_old_xface = Mfield_old[0]->array(mfi); // note M_old_xface include x,y,z components at |_x faces
        Array4<Real> const &M_old_yface = Mfield_old[1]->array(mfi); // note M_old_yface include x,y,z components at |_y faces
        Array4<Real> const &M_old_zface = Mfield_old[2]->array(mfi); // note M_old_zface include x,y,z components at |_z faces

        // macroscopic parameter
        amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);

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

        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto mag_parser = macroscopic_properties->m_mag_Ms_parser->compile<3>();

        amrex::Real const mu0_inv = 1. / PhysConst::mu0;

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Hx_stag, problo, dx, x, y, z);
                Real mag_Ms_arrx = mag_parser(x,y,z);
                if (mag_Ms_arrx == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrx = CoarsenIO::Interp( mu_arr, mu_stag, Hx_stag,
                                                             macro_cr, i, j, k, 0);
                    Hx(i, j, k) += 1. / mu_arrx * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                      - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                } else if (mag_Ms_arrx > 0){ // magnetic region
                    Hx(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                                                 - T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k));
                    if (coupling == 1) {
                        Hx(i, j, k) += - M_xface(i, j, k, 0) + M_old_xface(i, j, k, 0);
                    }
                }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Hy_stag, problo, dx, x, y, z);
                Real mag_Ms_arry = mag_parser(x,y,z);
                if (mag_Ms_arry == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arry = CoarsenIO::Interp( mu_arr, mu_stag, Hy_stag,
                                                             macro_cr, i, j, k, 0);
                    Hy(i, j, k) += 1. / mu_arry * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                      - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                } else if (mag_Ms_arry > 0){ // magnetic region
                    Hy(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                                                 - T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k));
                    if (coupling == 1){
                        Hy(i, j, k) += - M_yface(i, j, k, 1) + M_old_yface(i, j, k, 1);
                    }
                }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Hz_stag, problo, dx, x, y, z);
                Real mag_Ms_arrz = mag_parser(x,y,z);
                if (mag_Ms_arrz == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrz = CoarsenIO::Interp( mu_arr, mu_stag, Hz_stag,
                                                             macro_cr, i, j, k, 0);
                    Hz(i, j, k) += 1. / mu_arrz * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                      - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                } else if (mag_Ms_arrz > 0){ // magnetic region
                    Hz(i, j, k) += mu0_inv * dt * (T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                                                 - T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k));
                    if (coupling == 1){
                        Hz(i, j, k) += - M_zface(i, j, k, 2) + M_old_zface(i, j, k, 2);
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
        amrex::IntVect Bxnodal = Bfield[0]->ixType().toIntVect();
        amrex::IntVect Bynodal = Bfield[1]->ixType().toIntVect();
        amrex::IntVect Bznodal = Bfield[2]->ixType().toIntVect();
        Box const &tbx = mfi.tilebox(Bxnodal);
        Box const &tby = mfi.tilebox(Bynodal);
        Box const &tbz = mfi.tilebox(Bznodal);

        // macroscopic parameter
        amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);

        const auto dx = warpx.Geom(lev).CellSizeArray();
        const auto problo = warpx.Geom(lev).ProbLoArray();
        const auto mag_parser = macroscopic_properties->m_mag_Ms_parser->compile<3>();

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Bx_stag, problo, dx, x, y, z);
                Real mag_Ms_arrx = mag_parser(x,y,z);
                if (mag_Ms_arrx == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrx = CoarsenIO::Interp( mu_arr, mu_stag, Bx_stag,
                                                             macro_cr, i, j, k, 0);
                    Bx(i, j, k) = mu_arrx * Hx(i, j, k);
                } else if (mag_Ms_arrx > 0){
                    Bx(i, j, k) = PhysConst::mu0 * (M_xface(i, j, k, 0) + Hx(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, By_stag, problo, dx, x, y, z);
                Real mag_Ms_arry = mag_parser(x,y,z);
                if (mag_Ms_arry == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arry = CoarsenIO::Interp( mu_arr, mu_stag, By_stag,
                                                             macro_cr, i, j, k, 0);
                    By(i, j, k) =  mu_arry * Hy(i, j, k);
                } else if (mag_Ms_arry > 0){
                    By(i, j, k) = PhysConst::mu0 * (M_yface(i, j, k, 1) + Hy(i, j, k));
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, Bz_stag, problo, dx, x, y, z);
                Real mag_Ms_arrz = mag_parser(x,y,z);
                if (mag_Ms_arrz == 0._rt){ // nonmagnetic region
                    amrex::Real mu_arrz = CoarsenIO::Interp( mu_arr, mu_stag, Bz_stag,
                                                             macro_cr, i, j, k, 0);
                    Bz(i, j, k) = mu_arrz * Hz(i, j, k);
                } else if (mag_Ms_arrz > 0){
                    Bz(i, j, k) = PhysConst::mu0 * (M_zface(i, j, k, 2) + Hz(i, j, k));
                }
            });
    }
}
#endif // ifdef WARPX_MAG_LLG
#endif // ifndef WARPX_DIM_RZ
