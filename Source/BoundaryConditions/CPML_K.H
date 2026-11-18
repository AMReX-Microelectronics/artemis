/* Convolutional PML (CPML) helpers.
 *
 * The CPML uses the complex-frequency-shifted coordinate stretching
 *
 *     s_d = kappa_d + sigma_d / (alpha_d + i omega)       (d = x, y, z)
 *
 * (sigma_d and alpha_d are rates, i.e. sigma_SI/eps0 and alpha_SI/eps0), which turns
 * each spatial derivative into
 *
 *     d/dd  ->  (1/kappa_d) d/dd + psi_d
 *
 * where psi_d is the recursive convolution (Roden & Gedney, 2000)
 *
 *     psi_d^{new} = b_d psi_d^{old} + a_d (dF/dd),
 *     b_d = exp(-(sigma_d/kappa_d + alpha_d) dt),
 *     a_d = sigma_d / (kappa_d (sigma_d + kappa_d alpha_d)) (b_d - 1).
 *
 * Because the stretching only acts on the spatial derivatives, the material
 * response (eps, mu, sigma of the medium) is untouched: the E update keeps the
 * same alpha/beta (Lax-Wendroff or backward-Euler) coefficients as in the
 * interior, evaluated with the local eps(x) and sigma(x). This is what makes the
 * CPML reflectionless for arbitrary (and spatially varying) materials continued
 * into the PML, in contrast to formulations whose damping must be matched to eps.
 */
#ifndef WARPX_CPML_K_H_
#define WARPX_CPML_K_H_

#include "BoundaryConditions/PML.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>

#include <cmath>

/** Device-copyable view of the CPML 1D profiles of one SigmaBox, indexed by the
 *  physical direction (0=x, 1=y, 2=z). Directions that are not resolved by the
 *  simulation (y in 2D XZ) have null pointers and act as "no stretching". */
struct CPMLCoefs
{
    const amrex::Real* sig [3] = {nullptr, nullptr, nullptr}; // nodal positions
    const amrex::Real* sigs[3] = {nullptr, nullptr, nullptr}; // cell-centered positions
    const amrex::Real* kap [3] = {nullptr, nullptr, nullptr};
    const amrex::Real* kaps[3] = {nullptr, nullptr, nullptr};
    const amrex::Real* alp [3] = {nullptr, nullptr, nullptr};
    const amrex::Real* alps[3] = {nullptr, nullptr, nullptr};
    int lo[3] = {0, 0, 0};

    CPMLCoefs () = default;

    explicit CPMLCoefs (SigmaBox const& sb)
    {
#if defined(WARPX_DIM_3D)
        constexpr int dirs[AMREX_SPACEDIM] = {0, 1, 2};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        constexpr int dirs[AMREX_SPACEDIM] = {0, 2};
#else
        constexpr int dirs[AMREX_SPACEDIM] = {2};
#endif
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const int d = dirs[idim];
            sig [d] = sb.sigma[idim].data();
            sigs[d] = sb.sigma_star[idim].data();
            kap [d] = sb.kappa[idim].data();
            kaps[d] = sb.kappa_star[idim].data();
            alp [d] = sb.alpha[idim].data();
            alps[d] = sb.alpha_star[idim].data();
            lo  [d] = sb.sigma[idim].lo();
        }
    }
};

/** Staggering of a field component, indexed by physical direction (1 = nodal) */
inline amrex::GpuArray<int,3> cpml_stag (amrex::IndexType const& ixt)
{
    amrex::IntVect const iv = ixt.toIntVect();
#if defined(WARPX_DIM_3D)
    return {iv[0], iv[1], iv[2]};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return {iv[0], 0, iv[1]};
#else
    return {0, 0, iv[0]};
#endif
}

/** Grid index along physical direction dir */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int cpml_index (int dir, int i, int j, int k) noexcept
{
#if defined(WARPX_DIM_3D)
    return (dir == 0) ? i : ((dir == 1) ? j : k);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(k);
    return (dir == 0) ? i : j;
#else
    amrex::ignore_unused(dir, j, k);
    return i;
#endif
}

/** Coefficients (b, a, 1/kappa) of the CPML recursive convolution along direction
 *  dir, at grid index ii, for a field with nodal (1) or cell-centered (0) staggering
 *  along dir, for a time step dt. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cpml_coef (CPMLCoefs const& c, int dir, int nodal, int ii, amrex::Real dt,
                amrex::Real& b, amrex::Real& a, amrex::Real& inv_kappa) noexcept
{
    using namespace amrex::literals;
    if (c.sig[dir] == nullptr) {
        b = 1._rt; a = 0._rt; inv_kappa = 1._rt;
        return;
    }
    const int n = ii - c.lo[dir];
    const amrex::Real s  = nodal ? c.sig[dir][n] : c.sigs[dir][n];
    const amrex::Real kp = nodal ? c.kap[dir][n] : c.kaps[dir][n];
    const amrex::Real al = nodal ? c.alp[dir][n] : c.alps[dir][n];
    inv_kappa = 1._rt/kp;
    b = std::exp(-(s*inv_kappa + al)*dt);
    const amrex::Real denom = kp*(s + kp*al);
    a = (denom > 0._rt) ? s/denom*(b - 1._rt) : 0._rt;
}

/** Stretched derivative: if update_psi, first advances the memory variable psi
 *  (in place) over dt_psi; returns (1/kappa) dF + psi */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cpml_stretched_derivative (CPMLCoefs const& c, int dir, int nodal,
                                       int i, int j, int k, amrex::Real dt_psi, bool update_psi,
                                       amrex::Real dF, amrex::Real& psi) noexcept
{
    amrex::Real b, a, ik;
    cpml_coef(c, dir, nodal, cpml_index(dir, i, j, k), dt_psi, b, a, ik);
    if (update_psi) { psi = b*psi + a*dF; }
    return ik*dF + psi;
}

#endif // WARPX_CPML_K_H_
