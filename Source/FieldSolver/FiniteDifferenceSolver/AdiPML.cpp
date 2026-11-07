#include "AdiPML.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <string>

using namespace amrex;

namespace
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real adi_pml_rho (
        Real coord, Real inner_lo, Real inner_hi,
        int do_lo, int do_hi) noexcept
    {
        Real rho = 0._rt;
        if (do_lo && coord < inner_lo) {
            rho = inner_lo - coord;
        }
        if (do_hi && coord > inner_hi) {
            rho = amrex::max(rho, coord - inner_hi);
        }
        return rho;
    }

    // Grid-stretch factor along one axis: 1 inside [core_lo, core_hi], then growing by
    // `ratio` per computational cell up to kappa_max. Constant inside the PML (cells beyond
    // the inner edges), so the PML is uniform in physical cells.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real adi_stretch_kappa (
        Real s, Real ds, Real inner_lo, Real inner_hi,
        Real ratio, Real core_lo, Real core_hi, Real kappa_max) noexcept
    {
        Real const sc = amrex::max(inner_lo, amrex::min(inner_hi, s));
        Real const ncells = amrex::max(0._rt,
            amrex::max(core_lo - sc, sc - core_hi) / ds);
        return amrex::min(kappa_max, std::pow(ratio, ncells));
    }

    struct AdiCfsHost
    {
        GpuArray<Real, AMREX_SPACEDIM> problo{};
        GpuArray<Real, AMREX_SPACEDIM> probhi{};
        GpuArray<Real, AMREX_SPACEDIM> dx{};
        GpuArray<int, AMREX_SPACEDIM> domain_lo{};
        GpuArray<int, AMREX_SPACEDIM> n_cells{};
        GpuArray<int, AMREX_SPACEDIM> do_lo{};
        GpuArray<int, AMREX_SPACEDIM> do_hi{};
        GpuArray<Real, AMREX_SPACEDIM> sigma_max{};
        int ncell = 0;
        int on = 0;
        Real kappa_max = 1._rt;
        Real alpha_max = 0._rt;
        Real grade_m = 3._rt;
        Real dt_half = 0._rt;
        Real ep0 = 0._rt;
        GpuArray<Real, AMREX_SPACEDIM> stretch_ratio{};
        GpuArray<Real, AMREX_SPACEDIM> stretch_lo{};
        GpuArray<Real, AMREX_SPACEDIM> stretch_hi{};
        GpuArray<Real, AMREX_SPACEDIM> stretch_kappa_max{};
    };

    AdiCfsHost g_adi_cfs;
}

void
WarpX::GrowDomainForAdiOutsidePML ()
{
#if (AMREX_SPACEDIM != 3)
    return;
#else
    if (macroscopic_time_integrator_algo != MacroscopicTimeSteppingScheme::ADI) {
        return;
    }
    // In-domain mode: the user domain already includes the absorbing cells.
    if (do_pml_in_domain) {
        return;
    }

    IntVect grow_lo = IntVect::TheZeroVector();
    IntVect grow_hi = IntVect::TheZeroVector();
    bool need_grow = false;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (field_boundary_lo[idim] == FieldBoundaryType::PML) {
            grow_lo[idim] = pml_ncell;
            need_grow = true;
        }
        if (field_boundary_hi[idim] == FieldBoundaryType::PML) {
            grow_hi[idim] = pml_ncell;
            need_grow = true;
        }
    }
    if (!need_grow || pml_ncell <= 0) {
        return;
    }

    Geometry const& g0 = Geom(0);
    Box const domain0 = g0.Domain();
    RealBox rb = g0.ProbDomain();
    Real const* dx = g0.CellSize();

    IntVect ncell;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        ncell[idim] = domain0.length(idim);
        if (grow_lo[idim] > 0) {
            rb.setLo(idim, rb.lo(idim) - static_cast<Real>(grow_lo[idim]) * dx[idim]);
            ncell[idim] += grow_lo[idim];
        }
        if (grow_hi[idim] > 0) {
            rb.setHi(idim, rb.hi(idim) + static_cast<Real>(grow_hi[idim]) * dx[idim]);
            ncell[idim] += grow_hi[idim];
        }
    }

    IntVect const& bf = blockingFactor(0);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ncell[idim] % bf[idim] == 0,
            "ADI outside-domain PML: n_cell + pml_ncell must be divisible by "
            "amr.blocking_factor in each PML direction.");
    }

    Box const new_domain(IntVect::TheZeroVector(),
                         ncell - IntVect::TheUnitVector());
    Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(0, 0, 0)};
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        is_per[idim] = static_cast<int>(g0.isPeriodic(idim));
    }
    Geometry const gnew(new_domain, rb, g0.Coord(), is_per);
    SetGeometry(0, gnew);

    // Keep ParmParse metadata consistent with the grown domain.
    ParmParse pp_amr("amr");
    Vector<int> n_cell_arr(AMREX_SPACEDIM);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        n_cell_arr[idim] = ncell[idim];
    }
    pp_amr.addarr("n_cell", n_cell_arr);

    ParmParse pp_geom("geometry");
    Vector<Real> plo(AMREX_SPACEDIM), phi(AMREX_SPACEDIM);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        plo[idim] = rb.lo(idim);
        phi[idim] = rb.hi(idim);
    }
    pp_geom.addarr("prob_lo", plo);
    pp_geom.addarr("prob_hi", phi);

    amrex::Print() << "ADI CFS-PML: grew domain outside physical box by pml_ncell = "
                   << pml_ncell
                   << " (do_pml_in_domain = 0)\n";
#endif
}

void
WarpX::InitAdiPML ()
{
    adi_pml = 0;
    adi_pml_Lo = IntVect::TheZeroVector();
    adi_pml_Hi = IntVect::TheZeroVector();

#if (AMREX_SPACEDIM != 3)
    return;
#else
    if (macroscopic_time_integrator_algo != MacroscopicTimeSteppingScheme::ADI) {
        return;
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (field_boundary_lo[idim] == FieldBoundaryType::PML) {
            adi_pml_Lo[idim] = 1;
            adi_pml = 1;
        }
        if (field_boundary_hi[idim] == FieldBoundaryType::PML) {
            adi_pml_Hi[idim] = 1;
            adi_pml = 1;
        }
        // Grid stretching uses the same coordinate-factor machinery as the PML, independently of
        // the wall type: a stretched axis turns it on even with no PML wall (psi then stays zero).
        if (adi_stretch_ratio[idim] != 1._rt) {
            adi_pml = 1;
        }
    }
    if (!adi_pml) {
        return;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        finest_level == 0,
        "Macroscopic ADI PML is implemented only for a single AMR level.");
    // Do not construct split-field pml[lev]; FDTD PML stays off.
    do_pml = 0;

    BoxArray const& ba = boxArray(0);
    DistributionMapping const& dm = DistributionMap(0);

    // The PML profiles kappa(s), a(s), b(s) are evaluated analytically from g_adi_cfs (FillAdiCfsOnLayout);
    // the 3D adi_pml_stretch / adi_pml_a / adi_pml_b arrays were never read and are no longer allocated.

    auto const& eflag = std::array<IntVect, 3>{
        Efield_fp[0][0]->ixType().toIntVect(),
        Efield_fp[0][1]->ixType().toIntVect(),
        Efield_fp[0][2]->ixType().toIntVect()};
    auto const& bflag = std::array<IntVect, 3>{
        Bfield_fp[0][0]->ixType().toIntVect(),
        Bfield_fp[0][1]->ixType().toIntVect(),
        Bfield_fp[0][2]->ixType().toIntVect()};
    IntVect const& eng = Efield_fp[0][0]->nGrowVect();
    IntVect const& bng = Bfield_fp[0][0]->nGrowVect();

    AllocInitMultiFab(adi_psi_e[AdiPsiE::EXY],
        amrex::convert(ba, eflag[0]), dm, 1, eng, "adi_psi_e[exy]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EXZ],
        amrex::convert(ba, eflag[0]), dm, 1, eng, "adi_psi_e[exz]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EYX],
        amrex::convert(ba, eflag[1]), dm, 1, eng, "adi_psi_e[eyx]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EYZ],
        amrex::convert(ba, eflag[1]), dm, 1, eng, "adi_psi_e[eyz]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EZX],
        amrex::convert(ba, eflag[2]), dm, 1, eng, "adi_psi_e[ezx]", 0._rt);
    AllocInitMultiFab(adi_psi_e[AdiPsiE::EZY],
        amrex::convert(ba, eflag[2]), dm, 1, eng, "adi_psi_e[ezy]", 0._rt);

    AllocInitMultiFab(adi_psi_h[AdiPsiH::HXY],
        amrex::convert(ba, bflag[0]), dm, 1, bng, "adi_psi_h[hxy]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HXZ],
        amrex::convert(ba, bflag[0]), dm, 1, bng, "adi_psi_h[hxz]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HYX],
        amrex::convert(ba, bflag[1]), dm, 1, bng, "adi_psi_h[hyx]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HYZ],
        amrex::convert(ba, bflag[1]), dm, 1, bng, "adi_psi_h[hyz]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HZX],
        amrex::convert(ba, bflag[2]), dm, 1, bng, "adi_psi_h[hzx]", 0._rt);
    AllocInitMultiFab(adi_psi_h[AdiPsiH::HZY],
        amrex::convert(ba, bflag[2]), dm, 1, bng, "adi_psi_h[hzy]", 0._rt);

    FillAdiPmlProfiles();
    UpdateAdiPmlRecursionCoeffs(dt[0]);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (adi_stretch_ratio[idim] == 1._rt) { continue; }
        char const axis = "xyz"[idim];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            adi_stretch_ratio[idim] > 1._rt && adi_stretch_hi[idim] >= adi_stretch_lo[idim],
            "warpx.adi_<axis>stretch_ratio must exceed 1, with _lo <= _hi.");
        Real const ds = Geom(0).CellSize(idim);
        // Inner edge of each side: the PML inner edge on a PML side, the domain edge otherwise.
        Real const inner_lo = Geom(0).ProbLo(idim) + (adi_pml_Lo[idim] ? static_cast<Real>(pml_ncell) * ds : 0._rt);
        Real const inner_hi = Geom(0).ProbHi(idim) - (adi_pml_Hi[idim] ? static_cast<Real>(pml_ncell) * ds : 0._rt);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            adi_stretch_lo[idim] >= inner_lo - 1.e-6_rt * ds
            && adi_stretch_hi[idim] <= inner_hi + 1.e-6_rt * ds,
            "warpx.adi_<axis>stretch_lo and _hi must lie inside the domain (inside the PML inner edges on PML sides).");
        // Physical distance from the core to each PML inner edge: sum of ds*kappa over cells.
        Real below = 0._rt;
        Real above = 0._rt;
        int const ncell_s = static_cast<int>(std::lround((inner_hi - inner_lo) / ds));
        for (int n = 0; n < ncell_s; ++n) {
            Real const s = inner_lo + (static_cast<Real>(n) + 0.5_rt) * ds;
            Real const cell = ds * adi_stretch_kappa(s, ds, inner_lo, inner_hi,
                adi_stretch_ratio[idim], adi_stretch_lo[idim], adi_stretch_hi[idim], adi_stretch_kappa_max[idim]);
            if (s < adi_stretch_lo[idim]) { below += cell; }
            if (s > adi_stretch_hi[idim]) { above += cell; }
        }
        amrex::Print() << "ADI " << axis << " stretch: ratio = " << adi_stretch_ratio[idim]
                       << ", core " << adi_stretch_lo[idim] << " to " << adi_stretch_hi[idim]
                       << ", physical length below core = " << below
                       << ", above core = " << above
                       << ", kappa at the lo / hi inner edges = " << adi_stretch_kappa(inner_lo, ds, inner_lo, inner_hi,
                              adi_stretch_ratio[idim], adi_stretch_lo[idim], adi_stretch_hi[idim], adi_stretch_kappa_max[idim])
                       << " and " << adi_stretch_kappa(inner_hi, ds, inner_lo, inner_hi,
                              adi_stretch_ratio[idim], adi_stretch_lo[idim], adi_stretch_hi[idim], adi_stretch_kappa_max[idim])
                       << "\n";
    }

    amrex::Print() << "ADI CFS-PML: "
                   << (do_pml_in_domain ? "in-domain" : "outside-domain")
                   << ", pml_ncell = " << pml_ncell
                   << ", kappa_max = " << adi_pml_kappa_max
                   << ", alpha_max = " << adi_pml_alpha_max
                   << ", m = " << adi_pml_m
                   << "\n";
#endif
}

void
WarpX::FillAdiPmlProfiles ()
{
    // Kept for the call in InitAdiPML; the profiles are evaluated analytically where they are used.
}

void
WarpX::UpdateAdiPmlRecursionCoeffs (Real a_dt)
{
    g_adi_cfs.on = 0;
    if (!adi_pml) { return; }

    Real const dt_half = 0.5_rt * a_dt;
    auto const problo = Geom(0).ProbLoArray();
    auto const probhi = Geom(0).ProbHiArray();
    auto const dx = Geom(0).CellSizeArray();
    IntVect const domain_lo = Geom(0).Domain().smallEnd();
    int const ncell = pml_ncell;
    Real const alpha_max = adi_pml_alpha_max;
    Real const grade_m = adi_pml_m;
    Real const R = adi_pml_R;
    Real const sigma_max_user = adi_pml_sigma_max;
    IntVect const do_lo = adi_pml_Lo;
    IntVect const do_hi = adi_pml_Hi;
    Real const eta0 = std::sqrt(PhysConst::mu0 / PhysConst::ep0);
    Real const ep0 = PhysConst::ep0;

    g_adi_cfs.on = 1;
    g_adi_cfs.problo = problo;
    g_adi_cfs.probhi = probhi;
    g_adi_cfs.dx = dx;
    g_adi_cfs.ncell = ncell;
    g_adi_cfs.kappa_max = adi_pml_kappa_max;
    g_adi_cfs.alpha_max = alpha_max;
    g_adi_cfs.grade_m = grade_m;
    g_adi_cfs.dt_half = dt_half;
    g_adi_cfs.ep0 = ep0;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        g_adi_cfs.stretch_ratio[idim] = adi_stretch_ratio[idim];
        g_adi_cfs.stretch_lo[idim] = adi_stretch_lo[idim];
        g_adi_cfs.stretch_hi[idim] = adi_stretch_hi[idim];
        g_adi_cfs.stretch_kappa_max[idim] = adi_stretch_kappa_max[idim];
    }
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        g_adi_cfs.do_lo[idim] = do_lo[idim];
        g_adi_cfs.do_hi[idim] = do_hi[idim];
        g_adi_cfs.sigma_max[idim] = 0._rt;
        g_adi_cfs.domain_lo[idim] = domain_lo[idim];
        // Index origin and cell count are used to reject points outside the
        // domain while supporting a nonzero integer-domain lower bound.
        g_adi_cfs.n_cells[idim] = static_cast<int>(
            std::lround((probhi[idim] - problo[idim]) / dx[idim]));
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (!do_lo[idim] && !do_hi[idim]) { continue; }
        Real const pml_len = static_cast<Real>(ncell) * dx[idim];
        Real sigma_max = sigma_max_user;
        if (sigma_max < 0._rt && pml_len > 0._rt) {
            sigma_max = -(grade_m + 1._rt) * std::log(R) / (2._rt * eta0 * pml_len);
        }
        g_adi_cfs.sigma_max[idim] = sigma_max;
    }
}

void
FillAdiCfsOnLayout (MultiFab& dst, int dir, int quantity)
{
    // Sample at the Yee location of `dst` along `dir`, relative to the integer
    // domain origin. Nodal points sit at (idx-domain_lo)*Δs and cell-centered
    // points at (idx-domain_lo+1/2)*Δs.
    // quantity 3 is the coordinate factor 1/κ used in the field update.
    Real const fill = (quantity == 1) ? 0._rt : 1._rt;
    dst.setVal(fill);
    if (!g_adi_cfs.on || dir < 0 || dir >= AMREX_SPACEDIM) {
        return;
    }

    AdiCfsHost const p = g_adi_cfs;
    int const idir = dir;
    int const qty = quantity;
    int const domain_lo = p.domain_lo[idir];
    int const n_cells = p.n_cells[idir];
    int const nodal = dst.ixType()[idir];

    for (MFIter mfi(dst); mfi.isValid(); ++mfi) {
        auto const arr = dst.array(mfi);
        Box const bx = mfi.fabbox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const idx = (idir == 0) ? i : ((idir == 1) ? j : k);
            if (idx < domain_lo) { return; }
            if (nodal) {
                if (idx > domain_lo + n_cells) { return; }
            } else if (idx >= domain_lo + n_cells) {
                return;
            }
            Real const stagger = nodal ? 0._rt : 0.5_rt;
            Real const coord = p.problo[idir]
                + (static_cast<Real>(idx - domain_lo) + stagger) * p.dx[idir];
            Real kappa = 1._rt;
            Real aval = 0._rt;
            Real bval = 1._rt;
            int const lo_on = p.do_lo[idir];
            int const hi_on = p.do_hi[idir];
            Real const pml_len = static_cast<Real>(p.ncell) * p.dx[idir];
            // Inner edge of each side: the PML inner edge on a PML side, the domain edge otherwise.
            Real const inner_lo = p.problo[idir] + (lo_on ? pml_len : 0._rt);
            Real const inner_hi = p.probhi[idir] - (hi_on ? pml_len : 0._rt);
            // A stretched grid is d/ds -> (1/kappa_s) d/ds on the uniform grid, with or without PML.
            Real const kappa_s = (p.stretch_ratio[idir] != 1._rt)
                ? adi_stretch_kappa(coord, p.dx[idir], inner_lo, inner_hi,
                      p.stretch_ratio[idir], p.stretch_lo[idir], p.stretch_hi[idir], p.stretch_kappa_max[idir])
                : 1._rt;
            kappa = kappa_s;
            if (lo_on || hi_on) {
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho > 0._rt && pml_len > 0._rt) {
                    Real frac = rho / pml_len;
                    frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                    Real const poly = std::pow(frac, p.grade_m);
                    // s = kappa_s * (kappa + sigma_phys / (alpha + i w eps0)). The physical PML is
                    // kappa_s times thicker, so sigma_phys = sigma / kappa_s and the product
                    // kappa_s * sigma_phys is the sigma computed from the uniform-grid thickness.
                    kappa = kappa_s * (1._rt + (p.kappa_max - 1._rt) * poly);
                    Real const sig = p.sigma_max[idir] * poly;
                    Real const alp = p.alpha_max * (1._rt - frac);
                    Real const expo = -((sig / kappa) + alp) * p.dt_half / p.ep0;
                    bval = std::exp(expo);
                    Real const denom = kappa * (sig + kappa * alp);
                    aval = (std::abs(denom) > 0._rt)
                        ? (sig / denom) * (bval - 1._rt) : 0._rt;
                }
            }
            if (qty == 0) {
                arr(i,j,k) = kappa;
            } else if (qty == 1) {
                arr(i,j,k) = aval;
            } else if (qty == 2) {
                arr(i,j,k) = bval;
            } else {
                arr(i,j,k) = 1._rt / kappa;
            }
        });
    }
}
