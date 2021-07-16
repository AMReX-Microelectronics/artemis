#ifndef WARPX_MACROSCOPICPROPERTIES_H_
#define WARPX_MACROSCOPICPROPERTIES_H_

#include "MacroscopicProperties_fwd.H"

#include "Parser/WarpXParserWrapper.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>

/**
 * \brief This class contains the macroscopic properties of the medium needed to
 * evaluate macroscopic Maxwell equation.
 */
class
MacroscopicProperties
{
public:
     MacroscopicProperties (); // constructor
     /** Read user-defined macroscopic properties. Called in constructor. */
     void ReadParameters ();
     /** Initialize multifabs storing macroscopic multifabs */
     void InitData ();
     /** return MultiFab, sigma (conductivity) of the medium. */
     amrex::MultiFab& getsigma_mf  () {return (*m_sigma_mf);}
     /** return MultiFab, epsilon (permittivity) of the medium. */
     amrex::MultiFab& getepsilon_mf  () {return (*m_eps_mf);}
     /** return MultiFab, mu (permeability) of the medium. */
     amrex::MultiFab& getmu_mf  () {return (*m_mu_mf);}

     /** Initializes the Multifabs storing macroscopic properties
      *  with user-defined functions(x,y,z).
      */
     void InitializeMacroMultiFabUsingParser (amrex::MultiFab *macro_mf,
                                  HostDeviceParser<3> const& macro_parser, int lev);
     /** Gpu Vector with index type of the conductivity multifab */
     amrex::GpuArray<int, 3> sigma_IndexType;
     /** Gpu Vector with index type of the permittivity multifab */
     amrex::GpuArray<int, 3> epsilon_IndexType;
     /** Gpu Vector with index type of the permeability multifab */
     amrex::GpuArray<int, 3> mu_IndexType;
     /** Gpu Vector with index type of the Ex multifab */
     amrex::GpuArray<int, 3> Ex_IndexType;
     /** Gpu Vector with index type of the Ey multifab */
     amrex::GpuArray<int, 3> Ey_IndexType;
     /** Gpu Vector with index type of the Ez multifab */
     amrex::GpuArray<int, 3> Ez_IndexType;
#ifdef WARPX_MAG_LLG
     /** Gpu Vector with index type of the saturation magnetization multifab */
     amrex::GpuArray<int, 3> mag_Ms_IndexType;
     /** Gpu Vector with index type of the Gilbert damping factor multifab */
     amrex::GpuArray<int, 3> mag_alpha_IndexType;
     /** Gpu Vector with index type of the gyromagnetic ratio multifab */
     amrex::GpuArray<int, 3> mag_gamma_IndexType;
     /** Gpu Vector with index type of the Mx multifab */
     amrex::GpuArray<int, 3> Mx_IndexType;
     /** Gpu Vector with index type of the My multifab */
     amrex::GpuArray<int, 3> My_IndexType;
     /** Gpu Vector with index type of the Mz multifab */
     amrex::GpuArray<int, 3> Mz_IndexType;
     /** Gpu Vector with index type of the coefficient of the exchange coupling term multifab */
     amrex::GpuArray<int, 3> mag_exchange_IndexType;
     /** Gpu Vector with index type of the coefficient of the anisotropy coupling term multifab */
     amrex::GpuArray<int, 3> mag_anisotropy_IndexType;
     /** Gpu Vector of the anisotropy_axis for the anisotropy coupling term H_anisotropy in H_eff */
     amrex::GpuArray<amrex::Real, 3> mag_LLG_anisotropy_axis;
#endif
     /** Gpu Vector with index type of coarsening ratio with default value (1,1,1) */
     amrex::GpuArray<int, 3> macro_cr_ratio;

#ifdef WARPX_MAG_LLG
     amrex::MultiFab& getmag_Ms_mf () {return (*m_mag_Ms_mf);}
     amrex::MultiFab& getmag_alpha_mf () {return (*m_mag_alpha_mf);}
     amrex::MultiFab& getmag_gamma_mf () {return (*m_mag_gamma_mf);}
     amrex::MultiFab& getmag_exchange_mf () {return (*m_mag_exchange_mf);}
     amrex::MultiFab& getmag_anisotropy_mf () {return (*m_mag_anisotropy_mf);}

     amrex::Real getmag_normalized_error () {return m_mag_normalized_error;}
     int getmag_max_iter () {return m_mag_max_iter;}
     amrex::Real getmag_tol () {return m_mag_tol;}

     // interpolate the magnetic properties to B locations
     // magnetic properties are cell nodal
     // B locations are face centered
     // iv is an IntVect with a 1 in the face direction of interest, and 0 in the others
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real macro_avg_to_face (int i, int j, int k, amrex::IntVect iv, amrex::Array4<amrex::Real> const& macro_mag_prop){
         using namespace amrex;
         return ( 0.125_rt * ( macro_mag_prop(i        ,j        ,k        )
                             + macro_mag_prop(i-iv[0]+1,j        ,k        )
                             + macro_mag_prop(i        ,j-iv[1]+1,k        )
                             + macro_mag_prop(i        ,j        ,k-iv[2]+1)
                             + macro_mag_prop(i        ,j-iv[1]+1,k-iv[2]+1)
                             + macro_mag_prop(i-iv[0]+1,j        ,k-iv[2]+1)
                             + macro_mag_prop(i-iv[0]+1,j-iv[1]+1,k        )
                             + macro_mag_prop(i-iv[0]+1,j-iv[1]+1,k-iv[2]+1)
         ));
     };

     /** \brief
         * This GPU kernel averages a face-centered field onto a (possibly different nodality) face.
         * i,j,k are the spatial indices, n is the component index
         * iv_in is the nodality of the input field, iv_out is the nodality of the output.
         * Thus, if iv_in is the same as iv_out the value on the face is returned.
         * Some other examples when iv_in is different from iv_out:

         * Hy (0,1,0) components (in) to Hx face (1,0,0) (out)
         * return 0.25*(Hy(i,j,k)+Hy(i,j+1,k)+Hy(i-1,j,k)+Hy(i-1,j+1,k));
         *
         * \param[in] Fieldcomp  floating point data to be interpolated from one staggering location to another.
         * \param[in] iv_in           nodality of the input field, Fieldcomp
         * \param[in] iv_out         nodality of the output
         * \param[in] i                index along x of the Array4 Fieldcomp
         * \param[in] j                index along y of the Array4 Fieldcomp
         * \param[in] k                index along z of the Array4 Fieldcomp
     */
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real face_avg_to_face (int i, int j, int k, int n,
                                           amrex::IntVect iv_in, amrex::IntVect iv_out,
                                           amrex::Array4<amrex::Real> const& Fieldcomp) {
         using namespace amrex;
         return ( 0.125_rt * ( Fieldcomp(i                   , j                   , k                   , n)
                             + Fieldcomp(i+iv_in[0]-iv_out[0], j                   , k                   , n)
                             + Fieldcomp(i                   , j+iv_in[1]-iv_out[1], k                   , n)
                             + Fieldcomp(i                   , j                   , k+iv_in[2]-iv_out[2], n)
                             + Fieldcomp(i+iv_in[0]-iv_out[0], j+iv_in[1]-iv_out[1], k                   , n)
                             + Fieldcomp(i+iv_in[0]-iv_out[0], j                   , k+iv_in[2]-iv_out[2], n)
                             + Fieldcomp(i                   , j+iv_in[1]-iv_out[1], k+iv_in[2]-iv_out[2], n)
                             + Fieldcomp(i+iv_in[0]-iv_out[0], j+iv_in[1]-iv_out[1], k+iv_in[2]-iv_out[2], n)
                             ));
     };

     /** \brief
     * Calculate local H_maxwell = B / mu_o - M
     * \param[out] H_Maxwell local H_Maxwell field calculated using the magnetic constitutive relation
     * H_Maxwell is a scalar
     * \param[in] i x-index of the location under consideration
     * \param[in] j y-index of the location under consideration
     * \param[in] k z-index of the location under consideration
     * \param[in] n x, y, and z component of the H field we are interested in
     * \param[in] iv_in nodality of the input B field, Bcomp
     * \param[in] iv_out nodality of the output H_Maxwell
     * Note, we call face_avg_to_face on the B field with n=0 (B has only 1 component)
     * while Mcomp has n=0,1,2, indicating x,y,z components
     * note Mcomp has the same nodality as the final output H_maxwell
     */
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real getH_Maxwell (int i, int j, int k, int n,
                                   amrex::IntVect iv_in, amrex::IntVect iv_out,
                                   amrex::Array4<amrex::Real> const& Bcomp, amrex::Array4<amrex::Real> const& Mcomp) {
         using namespace amrex;
         amrex::Real H_Maxwell = face_avg_to_face(i, j, k, 0, iv_in, iv_out, Bcomp)/PhysConst::mu0
                                 - Mcomp(i, j, k, n); //magnetic constitutive relation
         return H_Maxwell;
     };

     /**
     update local M_field in the second-order time scheme
     the objective is to output component n of the M_field
     a and b have x,y,z components
     **/
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real updateM_field (int i, int j, int k, int n,
                                   amrex::Array4<amrex::Real> const& a, amrex::Array4<amrex::Real> const& b) {
         using namespace amrex;
         amrex::Real a_square = pow(a(i, j, k, 0), 2.0) + pow(a(i, j, k, 1), 2.0) + pow(a(i, j, k, 2), 2.0);
         amrex::Real a_dot_b =  a(i, j, k, 0) * b(i, j, k, 0) +
                                a(i, j, k, 1) * b(i, j, k, 1) +
                                a(i, j, k, 2) * b(i, j, k, 2);
         amrex::Real M_field;

     if(n==0){
         amrex::Real a_cross_b_x = a(i, j, k, 1) * b(i, j, k, 2) -
                                   a(i, j, k, 2) * b(i, j, k, 1);
         M_field = ( b(i, j, k, 0) + a_dot_b * a(i, j, k, 0) - a_cross_b_x ) / ( 1.0 + a_square);
     }
     else if(n==1){
         amrex::Real a_cross_b_y = a(i, j, k, 2) * b(i, j, k, 0) -
                                   a(i, j, k, 0) * b(i, j, k, 2);
         M_field = ( b(i, j, k, 1) + a_dot_b * a(i, j, k, 1) - a_cross_b_y ) / ( 1.0 + a_square);
     }
     else if(n==2){
         amrex::Real a_cross_b_z = a(i, j, k, 0) * b(i, j, k, 1) -
                                   a(i, j, k, 1) * b(i, j, k, 0);
         M_field = ( b(i, j, k, 2) + a_dot_b * a(i, j, k, 2) - a_cross_b_z ) / ( 1.0 + a_square);
     }
     else{
         printf("n=%d\n", n);
         amrex::Abort("Wrong component n of the M_field");
     };
         return M_field;
     };
#endif //closes ifdef MAG_LLG

     /** Conductivity, sigma, of the medium */
     amrex::Real m_sigma = 0.0;
     /** Permittivity, epsilon, of the medium */
     amrex::Real m_epsilon = PhysConst::ep0;
     /** Permeability, mu, of the medium */
     amrex::Real m_mu = PhysConst::mu0;

     /** Stores initialization type for conductivity : constant or parser */
     std::string m_sigma_s = "constant";
     /** Stores initialization type for permittivity : constant or parser */
     std::string m_epsilon_s = "constant";
     /** Stores initialization type for permeability : constant or parser */
     std::string m_mu_s = "constant";

     /** Parser Wrappers */
     // The ParserWrapper struct is constructed to safely use the GpuParser class
     // that can essentially be though of as a raw pointer. The GpuParser does
     // not have an explicit destructor and the AddPlasma subroutines handle the GpuParser
     // in a safe way. The ParserWrapper struct is used to avoid memory leak
     // in the EB parser functions.
     std::unique_ptr<ParserWrapper<3> > m_sigma_parser;
     std::unique_ptr<ParserWrapper<3> > m_epsilon_parser;
     std::unique_ptr<ParserWrapper<3> > m_mu_parser;
private:

#ifdef WARPX_MAG_LLG // preferred to use this tag multiple times for different variable types to keep formatting consistent
     /** Saturation magnetization, only applies for magnetic materials */
     amrex::Real m_mag_Ms;
     /** Gilbert damping, only applies for magnetic materials */
     amrex::Real m_mag_alpha;
     /** gyromagnetic ratio, should be a negative value, only applies for magnetic materials */
     amrex::Real m_mag_gamma;
     /** The coefficient of the exchange coupling term, only applies for magnetic materials */
     amrex::Real m_mag_exchange;
     /** The coefficient of the anisotropy coupling term, only applies for magnetic materials */
     amrex::Real m_mag_anisotropy;

     // If the magnitude of the magnetization deviates by more than this amount relative
     // to the user-defined Ms, abort.  Default 0.1.
     amrex::Real m_mag_normalized_error;

     // maximum iteration count for the second-order time advancement scheme of M field, default 100
     int m_mag_max_iter;

     // the relative tolerance for the second-order time advancement scheme of M field, default 0.0001
     amrex::Real m_mag_tol;

#endif

     /** Multifab for m_sigma */
     std::unique_ptr<amrex::MultiFab> m_sigma_mf;
     /** Multifab for m_epsilon */
     std::unique_ptr<amrex::MultiFab> m_eps_mf;
     /** Multifab for m_mu */
     std::unique_ptr<amrex::MultiFab> m_mu_mf;

#ifdef WARPX_MAG_LLG
     /** Multifab storing spatially varying saturation magnetization */
     std::unique_ptr<amrex::MultiFab> m_mag_Ms_mf;
     /** Multifab storing spatially varying Gilbert damping */
     std::unique_ptr<amrex::MultiFab> m_mag_alpha_mf;
     /** Multifab storing spatially varying gyromagnetic ratio */
     std::unique_ptr<amrex::MultiFab> m_mag_gamma_mf;
     /** Multifab storing spatially varying coefficient of the exchange coupling term */
     std::unique_ptr<amrex::MultiFab> m_mag_exchange_mf;
     /** Multifab storing spatially varying coefficient of the anisotropy coupling term */
     std::unique_ptr<amrex::MultiFab> m_mag_anisotropy_mf;
#endif


#ifdef WARPX_MAG_LLG
     std::string m_mag_Ms_s;
     std::string m_mag_alpha_s;
     std::string m_mag_gamma_s;
     std::string m_mag_exchange_s;
     std::string m_mag_anisotropy_s;
#endif

     /** string for storing parser function */
     std::string m_str_sigma_function;
     std::string m_str_epsilon_function;
     std::string m_str_mu_function;
#ifdef WARPX_MAG_LLG
     std::string m_str_mag_Ms_function;
     std::string m_str_mag_alpha_function;
     std::string m_str_mag_gamma_function;
     std::string m_str_mag_exchange_function;
     std::string m_str_mag_anisotropy_function;
#endif

     /** Parser Wrappers */
     // The ParserWrapper struct is constructed to safely use the GpuParser class
     // that can essentially be though of as a raw pointer. The GpuParser does
     // not have an explicit destructor and the AddPlasma subroutines handle the GpuParser
     // in a safe way. The ParserWrapper struct is used to avoid memory leak
     // in the EB parser functions.
#ifdef WARPX_MAG_LLG
     std::unique_ptr<ParserWrapper<3> > m_mag_Ms_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_alpha_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_gamma_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_exchange_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_anisotropy_parser;
#endif
};

/**
 * \brief
 * This struct contains only static functions to compute the co-efficients for the
 * Lax-Wendroff scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma), permittivity (epsilon).
 * Permeability of the material, mu, is used as (beta/mu) for the E-update
 * defined in MacroscopicEvolveECartesian().
 */
struct LaxWendroffAlgo {

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha (amrex::Real const sigma,
                               amrex::Real const epsilon,
                               amrex::Real dt) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt / epsilon;
         amrex::Real alpha = (1._rt - fac1)/(1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real beta (amrex::Real const sigma,
                              amrex::Real const epsilon,
                              amrex::Real dt) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt / epsilon;
         amrex::Real beta = dt / ( epsilon * (1._rt + fac1) );
         return beta;
     }

};

/**
 * \brief
 * This struct contains only static functions to compute the co-efficients for the
 * BackwardEuler scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma) and permittivity (epsilon).
 * Permeability of the material, mu, is used as (beta/mu) for the E-update
 * defined in MacroscopicEvolveECartesian().
 */
struct BackwardEulerAlgo {

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha (amrex::Real const sigma,
                               amrex::Real const epsilon,
                               amrex::Real dt) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt / epsilon;
         amrex::Real alpha = (1._rt)/(1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real beta (amrex::Real const sigma,
                              amrex::Real const epsilon,
                              amrex::Real dt) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt / epsilon;
         amrex::Real beta = dt / ( epsilon * (1._rt + fac1) );
         return beta;
     }

};

#endif // WARPX_MACROSCOPIC_PROPERTIES_H_
