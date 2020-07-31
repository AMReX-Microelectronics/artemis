#ifndef WARPX_MACROSCOPICPROPERTIES_H_
#define WARPX_MACROSCOPICPROPERTIES_H_


#include <AMReX_REAL.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>

#include <AMReX_MultiFab.H>
#include <algorithm>
#include <array>

#include "Parser/WarpXParser.H"
#include "Parser/WarpXParserWrapper.H"
#include "Utils/WarpXConst.H"

/**
 * \brief This class contains the macroscopic parameters of the medium needed to
 * evaluate macroscopic Maxwell equation.
 */
class
MacroscopicProperties
{
public:
     MacroscopicProperties (); // constructor
     /** \brief Read user-defined macroscopic properties. Called in constructor. */
     void ReadParameters ();
     /** Initialize multifabs storing macroscopic multifabs */
     void InitData ();

//     /** return Real, sigma (conductivity) of the medium. */
//     amrex::Real sigma () const noexcept {return m_sigma;}
//     /** return Real, epsilon (permittivity) of the medium. */
//     amrex::Real epsilon () const noexcept {return m_epsilon;}
//     /** return Real, mu (permeability) of the medium. */
//     amrex::Real mu () const noexcept {return m_mu;}
     /** */
     amrex::MultiFab& getsigma_mf  () {return (*m_sigma_mf);}
     amrex::MultiFab& getepsilon_mf  () {return (*m_eps_mf);}
     amrex::MultiFab& getmu_mf  () {return (*m_mu_mf);}

     void InitializeMacroMultiFabUsingParser (amrex::MultiFab *macro_mf,
                                  ParserWrapper<3> *macro_parser, int lev);

#ifdef WARPX_MAG_LLG
     amrex::MultiFab& getmag_Ms_mf () {return (*m_mag_Ms_mf);}
     amrex::MultiFab& getmag_alpha_mf () {return (*m_mag_alpha_mf);}
     amrex::MultiFab& getmag_gamma_mf () {return (*m_mag_gamma_mf);}

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

private:
     /** Conductivity, sigma, of the medium */
     amrex::Real m_sigma;
     /** Permittivity, epsilon, of the medium */
     amrex::Real m_epsilon;
     /** Permeability, mu, of the medium */
     amrex::Real m_mu;
#ifdef WARPX_MAG_LLG // preferred to use this tag multiple times for different variable types to keep formatting consistent
     /** Saturation magnetization, only applies for magnetic materials */
     amrex::Real m_mag_Ms;
     /** Gilbert damping, only applies for magnetic materials */
     amrex::Real m_mag_alpha;
     /** gyromagnetic ratio, should be a negative value, only applies for magnetic materials */
     amrex::Real m_mag_gamma;
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
#endif

     std::string m_sigma_s;
     std::string m_epsilon_s;
     std::string m_mu_s;
#ifdef WARPX_MAG_LLG
     std::string m_mag_Ms_s;
     std::string m_mag_alpha_s;
     std::string m_mag_gamma_s;
#endif

     /** string for storing parser function */
     std::string m_str_sigma_function;
     std::string m_str_epsilon_function;
     std::string m_str_mu_function;
#ifdef WARPX_MAG_LLG
     std::string m_str_mag_Ms_function;
     std::string m_str_mag_alpha_function;
     std::string m_str_mag_gamma_function;
#endif

     /** Parser Wrappers */
     // The ParserWrapper struct is constructed to safely use the GpuParser class
     // that can essentially be though of as a raw pointer. The GpuParser does
     // not have an explicit destructor and the AddPlasma subroutines handle the GpuParser
     // in a safe way. The ParserWrapper struct is used to avoid memory leak
     // in the EB parser functions.
     std::unique_ptr<ParserWrapper<3> > m_sigma_parser;
     std::unique_ptr<ParserWrapper<3> > m_epsilon_parser;
     std::unique_ptr<ParserWrapper<3> > m_mu_parser;
#ifdef WARPX_MAG_LLG
     std::unique_ptr<ParserWrapper<3> > m_mag_Ms_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_alpha_parser;
     std::unique_ptr<ParserWrapper<3> > m_mag_gamma_parser;
#endif
};

struct LaxWendroffAlgo {

     // iv(1,0,0) for cell-centered macroparameters to Ex edge
     // iv(0,1,0) for cell-centered macroparameters to Ey edge
     // iv(0,0,1) for cell-centered macroparameters to Ez edge
     // iv(0,1,1) for nodal macroparameters to Ex edge
     // iv(1,0,1) for nodal macroparameters to Ey edge
     // iv(1,1,0) for nodal macroparameters to Ez edge
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real macro_avg_to_edge (int i, int j, int k, amrex::IntVect iv,
                                           amrex::Array4<amrex::Real> const& macro_prop) {
         using namespace amrex;
         return ( 0.125_rt * ( macro_prop(i,         j,         k        )
                             + macro_prop(i+iv[0]-1, j,         k        )
                             + macro_prop(i,         j+iv[1]-1, k        )
                             + macro_prop(i,         j,         k+iv[2]-1)
                             + macro_prop(i+iv[0]-1, j+iv[1]-1, k        )
                             + macro_prop(i+iv[0]-1, j        , k+iv[2]-1)
                             + macro_prop(i,         j+iv[1]-1, k+iv[2]-1)
                             + macro_prop(i+iv[0]-1, j+iv[1]-1, k+iv[2]-1) ) );
     }

     /** */
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha (amrex::Array4<amrex::Real> const& sigma,
                               amrex::Array4<amrex::Real> const& epsilon,
                               amrex::Real dt, int i, int j, int k,
                               amrex::IntVect iv_cc) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * macro_avg_to_edge(i, j, k, iv_cc, sigma) * dt
                                   / macro_avg_to_edge(i, j, k, iv_cc, epsilon);
         amrex::Real alpha = (1._rt - fac1)/(1._rt + fac1);
         return alpha;
     };

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real beta (amrex::Array4<amrex::Real> const& sigma,
                              amrex::Array4<amrex::Real> const& epsilon,
                              amrex::Real dt, int i, int j, int k,
                              amrex::IntVect iv_cc) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * macro_avg_to_edge(i, j, k, iv_cc, sigma) * dt
                                   / macro_avg_to_edge(i, j, k, iv_cc, epsilon);
         amrex::Real beta = dt / ( macro_avg_to_edge(i, j, k, iv_cc, epsilon) * (1._rt + fac1) );
         return beta;
     };
};

struct BackwardEulerAlgo {

     // iv(1,0,0) for cell-centered macroparameters to Ex edge
     // iv(0,1,0) for cell-centered macroparameters to Ey edge
     // iv(0,0,1) for cell-centered macroparameters to Ez edge
     // iv(2,1,1) for nodal macroparameters to Ex edge
     // iv(1,2,1) for nodal macroparameters to Ey edge
     // iv(1,1,2) for nodal macroparameters to Ez edge
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real macro_avg_to_edge (int i, int j, int k, amrex::IntVect iv,
                                           amrex::Array4<amrex::Real> const& macro_prop) {
         using namespace amrex;
         return ( 0.125_rt * ( macro_prop(i,         j,         k        )
                             + macro_prop(i+iv[0]-1, j,         k        )
                             + macro_prop(i,         j+iv[1]-1, k        )
                             + macro_prop(i,         j,         k+iv[2]-1)
                             + macro_prop(i+iv[0]-1, j+iv[1]-1, k        )
                             + macro_prop(i+iv[0]-1, j        , k+iv[2]-1)
                             + macro_prop(i,         j+iv[1]-1, k+iv[2]-1)
                             + macro_prop(i+iv[0]-1, j+iv[1]-1, k+iv[2]-1) ) );
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha (amrex::Array4<amrex::Real> const& sigma,
                               amrex::Array4<amrex::Real> const& epsilon,
                               amrex::Real dt, int i, int j, int k,
                               amrex::IntVect iv_cc) {
         using namespace amrex;
         amrex::Real fac1 = macro_avg_to_edge(i, j, k, iv_cc, sigma) * dt
                            / macro_avg_to_edge(i, j, k, iv_cc, epsilon);
         amrex::Real alpha = 1._rt/ (1._rt + fac1);
         return alpha;
     };

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real beta (amrex::Array4<amrex::Real> const& sigma,
                              amrex::Array4<amrex::Real> const& epsilon,
                              amrex::Real dt, int i, int j, int k,
                              amrex::IntVect iv_cc) {
         using namespace amrex;
         amrex::Real fac1 = macro_avg_to_edge(i, j, k, iv_cc, sigma) * dt
                            / macro_avg_to_edge(i, j, k, iv_cc, epsilon);
         amrex::Real beta = dt / ( macro_avg_to_edge(i, j, k, iv_cc, epsilon) * (1._rt + fac1) );
         return beta;
     };
};

#endif // WARPX_MACROSCOPIC_PROPERTIES_H_
