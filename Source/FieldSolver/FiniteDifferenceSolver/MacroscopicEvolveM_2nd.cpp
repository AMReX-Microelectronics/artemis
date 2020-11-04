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

void FiniteDifferenceSolver::MacroscopicEvolveM_2nd(
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,      // Mfield contains three components MultiFab
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Bfield_old,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties){

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveMCartesian_2nd<CartesianYeeAlgorithm>(Mfield, H_biasfield, Bfield, Bfield_old, dt, macroscopic_properties);
    } else {
        amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
} // closes function MacroscopicEvolveM_2nd
#endif
#ifdef WARPX_MAG_LLG
template <typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveMCartesian_2nd(
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Bfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Bfield_old,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties){

    // obtain the maximum relative amount we let M deviate from Ms before aborting
    amrex::Real mag_normalized_error = macroscopic_properties->getmag_normalized_error();

    auto &warpx = WarpX::GetInstance();
    int coupling = warpx.mag_LLG_coupling;
    int M_normalization = warpx.mag_M_normalization;

    // build temporary vector<multifab,3> Mfield_prev, Mfield_error, a_temp, a_temp_static, b_temp_static
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_prev;   // M^n before the iteration
    std::array<std::unique_ptr<amrex::MultiFab>, 3> Mfield_error;  // The error of the M field between the twoiterations
    std::array<std::unique_ptr<amrex::MultiFab>, 3> a_temp;        // right-hand side of vector a, see the documentation
    std::array<std::unique_ptr<amrex::MultiFab>, 3> a_temp_static; // α M^n/|M| in the right-hand side of vector a, see the documentation
    std::array<std::unique_ptr<amrex::MultiFab>, 3> b_temp_static; // right-hand side of vector b, see the documentation

    // initialize Mfield_previous
    for (int i = 0; i < 3; i++){
        Mfield_prev[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        Mfield_error[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        Mfield_error[i]->setVal(0.); // initialize M_error to zero
        MultiFab::Copy(*Mfield_prev[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
    }
    // initialize a_temp, b_temp_static
    for (int i = 0; i < 3; i++){
        a_temp[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        a_temp_static[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        b_temp_static[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
    }

    // calculate the b_temp_static, a_temp_static
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
        Array4<Real> const &M_xface = Mfield[0]->array(mfi);      // note M_xface include x,y,z components at |_x faces
        Array4<Real> const &M_yface = Mfield[1]->array(mfi);      // note M_yface include x,y,z components at |_y faces
        Array4<Real> const &M_zface = Mfield[2]->array(mfi);      // note M_zface include x,y,z components at |_z faces
        Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
        Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
        Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
        Array4<Real> const &Bx_old = Bfield_old[0]->array(mfi);   // Bx is the x component at |_x faces
        Array4<Real> const &By_old = Bfield_old[1]->array(mfi);   // By is the y component at |_y faces
        Array4<Real> const &Bz_old = Bfield_old[2]->array(mfi);   // Bz is the z component at |_z faces

        // extract field data of a_temp_static and b_temp_static
        Array4<Real> const &a_temp_static_xface = a_temp_static[0]->array(mfi);
        Array4<Real> const &a_temp_static_yface = a_temp_static[1]->array(mfi);
        Array4<Real> const &a_temp_static_zface = a_temp_static[2]->array(mfi);
        Array4<Real> const &b_temp_static_xface = b_temp_static[0]->array(mfi);
        Array4<Real> const &b_temp_static_yface = b_temp_static[1]->array(mfi);
        Array4<Real> const &b_temp_static_zface = b_temp_static[2]->array(mfi);

        // extract stencil coefficients
        Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect()); /* just define which grid type */
        Box const &tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // loop over cells and update fields
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                Real mag_Ms_arrx    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);
                Real mag_alpha_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr);
                Real mag_gamma_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_gamma_arr);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrx > 0._rt)
                {
                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);

                    if (coupling == 1) {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Bx_old, M_xface);
                        Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), By_old, M_xface);
                        Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Bz_old, M_xface);
                    }

                    Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) : mag_Ms_arrx;
                    Real a_temp_static_coeff = mag_alpha_arrx / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                    Real b_temp_static_coeff = PhysConst::mu0 * mag_gamma_arrx / 2._rt;

                    // calculate a_temp_static_xface
                    // x component on x-faces of grid
                    a_temp_static_xface(i, j, k, 0) = a_temp_static_coeff * M_xface(i, j, k, 0);

                    // y component on x-faces of grid
                    a_temp_static_xface(i, j, k, 1) = a_temp_static_coeff * M_xface(i, j, k, 1);

                    // z component on x-faces of grid
                    a_temp_static_xface(i, j, k, 2) = a_temp_static_coeff * M_xface(i, j, k, 2);

                    // calculate b_temp_static_xface
                    // x component on x-faces of grid
                    b_temp_static_xface(i, j, k, 0) = M_xface(i, j, k, 0) + dt * b_temp_static_coeff * (M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff);

                    // y component on x-faces of grid
                    b_temp_static_xface(i, j, k, 1) = M_xface(i, j, k, 1) + dt * b_temp_static_coeff * (M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff);

                    // z component on x-faces of grid
                    b_temp_static_xface(i, j, k, 2) = M_xface(i, j, k, 2) + dt * b_temp_static_coeff * (M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff);
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                Real mag_Ms_arry    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);
                Real mag_alpha_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr);
                Real mag_gamma_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_gamma_arr);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arry > 0._rt)
                {
                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);

                    if (coupling == 1) {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Bx_old, M_yface);
                        Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), By_old, M_yface);
                        Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Bz_old, M_yface);
                    }

                    Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) : mag_Ms_arry;
                    Real a_temp_static_coeff = mag_alpha_arry / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                    Real b_temp_static_coeff = PhysConst::mu0 * mag_gamma_arry / 2._rt;

                    // calculate a_temp_static_yface
                    // x component on y-faces of grid
                    a_temp_static_yface(i, j, k, 0) = a_temp_static_coeff * M_yface(i, j, k, 0);

                    // y component on y-faces of grid
                    a_temp_static_yface(i, j, k, 1) = a_temp_static_coeff * M_yface(i, j, k, 1);

                    // z component on y-faces of grid
                    a_temp_static_yface(i, j, k, 2) = a_temp_static_coeff * M_yface(i, j, k, 2);

                    // calculate b_temp_static_yface
                    // x component on y-faces of grid
                    b_temp_static_yface(i, j, k, 0) = M_yface(i, j, k, 0) + dt * b_temp_static_coeff * (M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff);

                    // y component on y-faces of grid
                    b_temp_static_yface(i, j, k, 1) = M_yface(i, j, k, 1) + dt * b_temp_static_coeff * (M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff);

                    // z component on y-faces of grid
                    b_temp_static_yface(i, j, k, 2) = M_yface(i, j, k, 2) + dt * b_temp_static_coeff * (M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff);
                }
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                Real mag_Ms_arrz    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);
                Real mag_alpha_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr);
                Real mag_gamma_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_gamma_arr);

                // determine if the material is nonmagnetic or not
                if (mag_Ms_arrz > 0._rt)
                {
                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                    if (coupling == 1) {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Bx_old, M_zface);
                        Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), By_old, M_zface);
                        Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Bz_old, M_zface);
                    }

                    Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) : mag_Ms_arrz;
                    Real a_temp_static_coeff = mag_alpha_arrz / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                    Real b_temp_static_coeff = PhysConst::mu0 * mag_gamma_arrz / 2._rt;

                    // calculate a_temp_static_zface
                    // x component on z-faces of grid
                    a_temp_static_zface(i, j, k, 0) = a_temp_static_coeff * M_zface(i, j, k, 0);

                    // y component on z-faces of grid
                    a_temp_static_zface(i, j, k, 1) = a_temp_static_coeff * M_zface(i, j, k, 1);

                    // z component on z-faces of grid
                    a_temp_static_zface(i, j, k, 2) = a_temp_static_coeff * M_zface(i, j, k, 2);

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
        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){
            auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
            auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
            auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
            // extract material properties
            Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
            Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
            Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);

            // extract field data
            Array4<Real> const &M_xface = Mfield[0]->array(mfi);      // note M_xface include x,y,z components at |_x faces
            Array4<Real> const &M_yface = Mfield[1]->array(mfi);      // note M_yface include x,y,z components at |_y faces
            Array4<Real> const &M_zface = Mfield[2]->array(mfi);      // note M_zface include x,y,z components at |_z faces
            Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
            Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
            Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
            Array4<Real> const &Bx = Bfield[0]->array(mfi);           // Bx is the x component at |_x faces
            Array4<Real> const &By = Bfield[1]->array(mfi);           // By is the y component at |_y faces
            Array4<Real> const &Bz = Bfield[2]->array(mfi);           // Bz is the z component at |_z faces

            // extract field data of Mfield_prev, Mfield_error, a_temp, a_temp_static, and b_temp_static
            Array4<Real> const &M_prev_xface = Mfield_prev[0]->array(mfi);
            Array4<Real> const &M_prev_yface = Mfield_prev[1]->array(mfi);
            Array4<Real> const &M_prev_zface = Mfield_prev[2]->array(mfi);
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

            // extract stencil coefficients
            Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const &tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    Real mag_Ms_arrx    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);
                    Real mag_alpha_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr);
                    Real mag_gamma_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_gamma_arr);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arrx > 0._rt)
                    {
                        // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                        Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                        Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);

                        if (coupling == 1) {
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Bx, M_prev_xface);
                            Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), By, M_prev_xface);
                            Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Bz, M_prev_xface);
                        }

                        // calculate the a_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                        Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrx) / 2._rt;

                        // calculate a_temp_xface
                        // x component on x-faces of grid
                        a_temp_xface(i, j, k, 0) = -(dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_xface(i, j, k, 0));

                        // y component on x-faces of grid
                        a_temp_xface(i, j, k, 1) = -(dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_xface(i, j, k, 1));

                        // z component on x-faces of grid
                        a_temp_xface(i, j, k, 2) = -(dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_xface(i, j, k, 2));

                        // update M_xface from a and b using the updateM_field
                        // x component on x-faces of grid
                        M_xface(i, j, k, 0) = MacroscopicProperties::updateM_field(i, j, k, 0, a_temp_xface, b_temp_static_xface);

                        // y component on x-faces of grid
                        M_xface(i, j, k, 1) = MacroscopicProperties::updateM_field(i, j, k, 1, a_temp_xface, b_temp_static_xface);

                        // z component on x-faces of grid
                        M_xface(i, j, k, 2) = MacroscopicProperties::updateM_field(i, j, k, 2, a_temp_xface, b_temp_static_xface);

                        // temporary normalized magnitude of M_xface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) +
                                                                       std::pow(M_xface(i, j, k, 2), 2._rt)) / mag_Ms_arrx;

                        if (M_normalization == 1) {
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                                amrex::Abort("Exceed the normalized error of the M_xface field");
                            }
                            // normalize the M_xface field
                            M_xface(i, j, k, 0) /= M_magnitude_normalized;
                            M_xface(i, j, k, 1) /= M_magnitude_normalized;
                            M_xface(i, j, k, 2) /= M_magnitude_normalized;
                        } else if (M_normalization == 0) {
                            // check the normalized error
                            if (M_magnitude_normalized > 1._rt + mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrx);
                                amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                            } else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error) {
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_xface
                        // x component on x-faces of grid
                        M_error_xface(i, j, k, 0) = amrex::Math::abs((M_xface(i, j, k, 0) - M_prev_xface(i, j, k, 0))) / mag_Ms_arrx;

                        // y component on x-faces of grid
                        M_error_xface(i, j, k, 1) = amrex::Math::abs((M_xface(i, j, k, 1) - M_prev_xface(i, j, k, 1))) / mag_Ms_arrx;

                        // z component on x-faces of grid
                        M_error_xface(i, j, k, 2) = amrex::Math::abs((M_xface(i, j, k, 2) - M_prev_xface(i, j, k, 2))) / mag_Ms_arrx;
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    Real mag_Ms_arry    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);
                    Real mag_alpha_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr);
                    Real mag_gamma_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_gamma_arr);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arry > 0._rt)
                    {
                        // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                        Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                        Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);

                        if (coupling == 1) {
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Bx, M_prev_yface);
                            Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), By, M_prev_yface);
                            Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Bz, M_prev_yface);
                        }

                        // calculate the a_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                        Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arry) / 2._rt;

                        // calculate a_temp_yface
                        // x component on y-faces of grid
                        a_temp_yface(i, j, k, 0) = -(dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_yface(i, j, k, 0));

                        // y component on y-faces of grid
                        a_temp_yface(i, j, k, 1) = -(dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_yface(i, j, k, 1));

                        // z component on y-faces of grid
                        a_temp_yface(i, j, k, 2) = -(dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_yface(i, j, k, 2));

                        // update M_yface from a and b using the updateM_field
                        // x component on y-faces of grid
                        M_yface(i, j, k, 0) = MacroscopicProperties::updateM_field(i, j, k, 0, a_temp_yface, b_temp_static_yface);

                        // y component on y-faces of grid
                        M_yface(i, j, k, 1) = MacroscopicProperties::updateM_field(i, j, k, 1, a_temp_yface, b_temp_static_yface);

                        // z component on y-faces of grid
                        M_yface(i, j, k, 2) = MacroscopicProperties::updateM_field(i, j, k, 2, a_temp_yface, b_temp_static_yface);

                        // temporary normalized magnitude of M_yface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) +
                                                                       std::pow(M_yface(i, j, k, 2), 2._rt)) / mag_Ms_arry;

                        if (M_normalization == 1) {
                            // check the normalized error
                            if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                                amrex::Abort("Exceed the normalized error of the M_yface field");
                            }
                            // normalize the M_yface field
                            M_yface(i, j, k, 0) /= M_magnitude_normalized;
                            M_yface(i, j, k, 1) /= M_magnitude_normalized;
                            M_yface(i, j, k, 2) /= M_magnitude_normalized;
                        } else if (M_normalization == 0) {
                            // check the normalized error
                            if (M_magnitude_normalized > 1._rt + mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arry);
                                amrex::Abort("Caution: Unsaturated material has M_yface exceeding the saturation magnetization");
                            } else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error) {
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_yface
                        // x component on y-faces of grid
                        M_error_yface(i, j, k, 0) = amrex::Math::abs((M_yface(i, j, k, 0) - M_prev_yface(i, j, k, 0))) / mag_Ms_arry;

                        // y component on y-faces of grid
                        M_error_yface(i, j, k, 1) = amrex::Math::abs((M_yface(i, j, k, 1) - M_prev_yface(i, j, k, 1))) / mag_Ms_arry;

                        // z component on y-faces of grid
                        M_error_yface(i, j, k, 2) = amrex::Math::abs((M_yface(i, j, k, 2) - M_prev_yface(i, j, k, 2))) / mag_Ms_arry;
                    }
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    Real mag_Ms_arrz    = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);
                    Real mag_alpha_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr);
                    Real mag_gamma_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_gamma_arr);

                    // determine if the material is nonmagnetic or not
                    if (mag_Ms_arrz > 0._rt)
                    {
                        // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                        Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                        Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                        if (coupling == 1) {
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                            // H_maxwell
                            Hx_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Bx, M_prev_zface);
                            Hy_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), By, M_prev_zface);
                            Hz_eff += MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Bz, M_prev_zface);
                        }

                        // calculate the a_temp_static_coeff (it is divided by 2.0 because the input dt is actually dt/2.0)
                        Real a_temp_dynamic_coeff = PhysConst::mu0 * amrex::Math::abs(mag_gamma_arrz) / 2._rt;

                        // calculate a_temp_zface
                        // x component on z-faces of grid
                        a_temp_zface(i, j, k, 0) = -(dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_zface(i, j, k, 0));

                        // y component on z-faces of grid
                        a_temp_zface(i, j, k, 1) = -(dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_zface(i, j, k, 1));

                        // z component on z-faces of grid
                        a_temp_zface(i, j, k, 2) = -(dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_zface(i, j, k, 2));

                        // update M_zface from a and b using the updateM_field
                        // x component on z-faces of grid
                        M_zface(i, j, k, 0) = MacroscopicProperties::updateM_field(i, j, k, 0, a_temp_zface, b_temp_static_zface);

                        // y component on z-faces of grid
                        M_zface(i, j, k, 1) = MacroscopicProperties::updateM_field(i, j, k, 1, a_temp_zface, b_temp_static_zface);

                        // z component on z-faces of grid
                        M_zface(i, j, k, 2) = MacroscopicProperties::updateM_field(i, j, k, 2, a_temp_zface, b_temp_static_zface);

                        // temporary normalized magnitude of M_zface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) +
                                                                       std::pow(M_zface(i, j, k, 2), 2._rt)) / mag_Ms_arrz;

                        if (M_normalization == 1) {
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                                amrex::Abort("Exceed the normalized error of the M_zface field");
                            }
                            // normalize the M_zface field
                            M_zface(i, j, k, 0) /= M_magnitude_normalized;
                            M_zface(i, j, k, 1) /= M_magnitude_normalized;
                            M_zface(i, j, k, 2) /= M_magnitude_normalized;
                        } else if (M_normalization == 0) {
                            // check the normalized error
                            if (M_magnitude_normalized > 1._rt + mag_normalized_error) {
                                printf("i = %d, j=%d, k=%d\n", i, j, k);
                                printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized, mag_Ms_arrz);
                                amrex::Abort("Caution: Unsaturated material has M_zface exceeding the saturation magnetization");
                            } else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error) {
                                // normalize the M_zface field
                                M_zface(i, j, k, 0) /= M_magnitude_normalized;
                                M_zface(i, j, k, 1) /= M_magnitude_normalized;
                                M_zface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error_zface
                        // x component on z-faces of grid
                        M_error_zface(i, j, k, 0) = amrex::Math::abs((M_zface(i, j, k, 0) - M_prev_zface(i, j, k, 0))) / mag_Ms_arrz;

                        // y component on z-faces of grid
                        M_error_zface(i, j, k, 1) = amrex::Math::abs((M_zface(i, j, k, 1) - M_prev_zface(i, j, k, 1))) / mag_Ms_arrz;

                        // z component on z-faces of grid
                        M_error_zface(i, j, k, 2) = amrex::Math::abs((M_zface(i, j, k, 2) - M_prev_zface(i, j, k, 2))) / mag_Ms_arrz;
                    }
                });
        }

        // Check the error between Mfield and Mfield_prev and decide whether another iteration is needed
        amrex::Real M_iter_maxerror = -1._rt;
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 3; j++) {
                Real M_iter_error = Mfield_error[i]->norm0(j);
                if (M_iter_error >= M_iter_maxerror) {
                    M_iter_maxerror = M_iter_error;
                }
            }
        }

        if (M_iter_maxerror <= M_tol) {

            stop_iter = 1;

            // normalize M
            if (M_normalization == 2)
            {

                for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){
                    auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
                    // extract material properties
                    Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);

                    // extract field data
                    Array4<Real> const &M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
                    Array4<Real> const &M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
                    Array4<Real> const &M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces

                    // extract tileboxes for which to loop
                    Box const &tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect()); /* just define which grid type */
                    Box const &tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
                    Box const &tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

                    // loop over cells and update fields
                    amrex::ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arrx = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

                            if (mag_Ms_arrx > 0._rt)
                            {
                                // temporary normalized magnitude of M_xface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_xface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arrx;

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
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arry = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);

                            if (mag_Ms_arry > 0._rt)
                            {
                                // temporary normalized magnitude of M_yface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_yface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arry;

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
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            Real mag_Ms_arrz = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);

                            if (mag_Ms_arrz > 0._rt)
                            {
                                // temporary normalized magnitude of M_zface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) +
                                                                               std::pow(M_zface(i, j, k, 2), 2._rt)) /
                                                                     mag_Ms_arrz;

                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > mag_normalized_error)
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
                        });
                }
            }
        } else {
            // Copy Mfield to Mfield_previous
            for (int i = 0; i < 3; i++) {
                MultiFab::Copy(*Mfield_prev[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
            }
        }

        if (M_iter >= M_max_iter) {
            amrex::Abort("The M_iter exceeds the M_max_iter");
            amrex::Print() << "The M_iter = " << M_iter << " exceeds the M_max_iter = " << M_max_iter << std::endl;
        } else {
            M_iter++;
            amrex::Print() << "Finish " << M_iter << " times iteration with M_iter_maxerror = " << M_iter_maxerror << " and M_tol = " << M_tol << std::endl;
        }

    } // end the iteration
}
#endif
