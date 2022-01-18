#ifndef WARPX_CELLCENTERFUNCTOR_H_
#define WARPX_CELLCENTERFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF and store result in mf_out.
 */
class
CellCenterFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mf_src source multifab.
     * \param[in] lev level of multifab. Used for averaging in rz.
     * \param[in] crse_ratio for interpolating field values from the simulation MultiFab, src_mf,
                  to the output diagnostic MultiFab, mf_dst.
     * \param[in] convertRZmodes2cartesian (in cylindrical) whether to
     *            sum all modes in mf_src before cell-centering into dst multifab.
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     * \param[in] scomp starting component of mf_src from which values are
                  averaged/interpolated to mf_dst
     */
    CellCenterFunctor(const amrex::MultiFab * const mf_src, const int lev,
                      const amrex::IntVect crse_ratio,
                      bool convertRZmodes2cartesian=true, int ncomp=1, int scomp=0);
    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    virtual void operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer=0*/) const override;

     /** \brief return m_scomp
      * m_scomp is the starting component of mf_src from which values are
      * averaged/interpolated to mf_dst
      */
    int sComp () const { return m_scomp; }

private:
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const * const m_mf_src = nullptr;
    int m_lev; /**< level on which mf_src is defined (used in cylindrical) */
    /**< (for cylindrical) whether to average all modes into 1 comp */
    bool m_convertRZmodes2cartesian;
    /** starting component of mf_src from which values are averaged/interpolated to mf_dst */
    int m_scomp;
};

#endif // WARPX_CELLCENTERFUNCTOR_H_
