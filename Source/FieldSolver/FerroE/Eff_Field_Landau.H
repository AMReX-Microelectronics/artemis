#include "FerroE.H"

namespace landau_coefficient{

    // Gradient energy coefficients
    constexpr amrex::Real alpha_1 = -3.7116e7;
    constexpr amrex::Real alpha_11 = -2.097e8;
    constexpr amrex::Real alpha_12 = 7.974e8;
    constexpr amrex::Real alpha_111 = 1.294e9;
    constexpr amrex::Real alpha_112 = -1.95e9;
    constexpr amrex::Real alpha_123 = -2.5e9;
    constexpr amrex::Real alpha_1111 = 3.863e10;
    constexpr amrex::Real alpha_1112 = 2.529e10;
    constexpr amrex::Real alpha_1122 = 1.637e10;
    constexpr amrex::Real alpha_1123 = 1.367e10;
}

//Compute x-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ex_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    using namespace landau_coefficient;
    return -2.*alpha_1 * Px 
           -4.*alpha_11 * std::pow(Px,3.) -2.*alpha_12 * Px * (std::pow(Py,2.) + std::pow(Pz,2.)) 
           -6.*alpha_111 * std::pow(Px,5.) 
           -4.*alpha_112 * std::pow(Px,3.) * (std::pow(Py,2.) + std::pow(Pz,2.)) 
           -2.*alpha_112 * Px * (std::pow(Py,4.) + std::pow(Pz,4.)) 
           -2.*alpha_123 * Px * std::pow(Py,2.) * std::pow(Pz,2.)
           -8.*alpha_1111 * std::pow(Px,7.) 
           -2.*alpha_1112 * Px * ( std::pow(Py,6.) + std::pow(Pz,6.) + 3.*std::pow(Px,4.)*(std::pow(Py,2) + std::pow(Pz,2))) 
           -4.*alpha_1122*std::pow(Px,3.)*(std::pow(Py,4.) + std::pow(Pz,4)) 
           -2.*alpha_1123*Px*std::pow(Py,2.)*std::pow(Pz,2.)*(2.*std::pow(Px,2.) + std::pow(Py,2.) + std::pow(Pz,2));
}


//Compute y-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ey_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    using namespace landau_coefficient;
    return -2.*alpha_1 * Py 
           -4.*alpha_11 * std::pow(Py,3.) -2.*alpha_12 * Py * (std::pow(Pz,2.) + std::pow(Px,2.)) 
           -6.*alpha_111 * std::pow(Py,5.) 
           -4.*alpha_112 * std::pow(Py,3.) * (std::pow(Pz,2.) + std::pow(Px,2.)) 
           -2.*alpha_112 * Py * (std::pow(Pz,4.) + std::pow(Px,4.)) 
           -2.*alpha_123 * Py * std::pow(Pz,2.) * std::pow(Px,2.)
           -8.*alpha_1111 * std::pow(Py,7.) 
           -2.*alpha_1112 * Py * ( std::pow(Pz,6.) + std::pow(Px,6.) + 3.*std::pow(Py,4.)*(std::pow(Pz,2) + std::pow(Px,2))) 
           -4.*alpha_1122*std::pow(Py,3.)*(std::pow(Pz,4.) + std::pow(Px,4)) 
           -2.*alpha_1123*Py*std::pow(Pz,2.)*std::pow(Px,2.)*(2.*std::pow(Py,2.) + std::pow(Pz,2.) + std::pow(Px,2));
}


//Compute z-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ez_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    using namespace landau_coefficient;
    return -2.*alpha_1 * Pz 
           -4.*alpha_11 * std::pow(Pz,3.) -2.*alpha_12 * Pz * (std::pow(Px,2.) + std::pow(Py,2.)) 
           -6.*alpha_111 * std::pow(Pz,5.) 
           -4.*alpha_112 * std::pow(Pz,3.) * (std::pow(Px,2.) + std::pow(Py,2.)) 
           -2.*alpha_112 * Pz * (std::pow(Px,4.) + std::pow(Py,4.)) 
           -2.*alpha_123 * Pz * std::pow(Px,2.) * std::pow(Py,2.)
           -8.*alpha_1111 * std::pow(Pz,7.) 
           -2.*alpha_1112 * Pz * ( std::pow(Px,6.) + std::pow(Py,6.) + 3.*std::pow(Pz,4.)*(std::pow(Px,2) + std::pow(Py,2))) 
           -4.*alpha_1122*std::pow(Pz,3.)*(std::pow(Px,4.) + std::pow(Py,4)) 
           -2.*alpha_1123*Pz*std::pow(Px,2.)*std::pow(Py,2.)*(2.*std::pow(Pz,2.) + std::pow(Px,2.) + std::pow(Py,2));
}

