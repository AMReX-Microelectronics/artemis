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
    IntVect const ng = IntVect(1);

    for (int idim = 0; idim < 3; ++idim) {
        AllocInitMultiFab(
            adi_pml_stretch[idim], ba, dm, 1, ng,
            "adi_pml_stretch[" + std::to_string(idim) + "]", 1._rt);
        AllocInitMultiFab(
            adi_pml_a[idim], ba, dm, 1, ng,
            "adi_pml_a[" + std::to_string(idim) + "]", 0._rt);
        AllocInitMultiFab(
            adi_pml_b[idim], ba, dm, 1, ng,
            "adi_pml_b[" + std::to_string(idim) + "]", 1._rt);
    }

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
    if (!adi_pml) { return; }

    auto const problo = Geom(0).ProbLoArray();
    auto const probhi = Geom(0).ProbHiArray();
    auto const dx = Geom(0).CellSizeArray();
    IntVect const domain_lo = Geom(0).Domain().smallEnd();
    int const ncell = pml_ncell;
    Real const kappa_max = adi_pml_kappa_max;
    Real const grade_m = adi_pml_m;
    IntVect const do_lo = adi_pml_Lo;
    IntVect const do_hi = adi_pml_Hi;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        MultiFab& kappa_mf = *adi_pml_stretch[idim];
        kappa_mf.setVal(1._rt);
        int const idir = idim;
        int const lo_on = do_lo[idim];
        int const hi_on = do_hi[idim];
        if (!lo_on && !hi_on) { continue; }

        for (MFIter mfi(kappa_mf); mfi.isValid(); ++mfi) {
            auto const arr = kappa_mf.array(mfi);
            Box const bx = mfi.fabbox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const idx = (idir == 0) ? i : ((idir == 1) ? j : k);
                Real const coord =
                    problo[idir]
                    + (static_cast<Real>(idx - domain_lo[idir]) + 0.5_rt)
                        * dx[idir];
                Real const pml_len = static_cast<Real>(ncell) * dx[idir];
                Real const inner_lo = problo[idir] + pml_len;
                Real const inner_hi = probhi[idir] - pml_len;
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho <= 0._rt || pml_len <= 0._rt) {
                    arr(i,j,k) = 1._rt;
                    return;
                }
                Real frac = rho / pml_len;
                frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                Real const poly = std::pow(frac, grade_m);
                arr(i,j,k) = 1._rt + (kappa_max - 1._rt) * poly;
            });
        }
        adi_pml_stretch[idim]->setBndry(1._rt);
        adi_pml_stretch[idim]->FillBoundary(Geom(0).periodicity());
    }
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
        MultiFab& a_mf = *adi_pml_a[idim];
        MultiFab& b_mf = *adi_pml_b[idim];
        MultiFab const& kappa_mf = *adi_pml_stretch[idim];
        a_mf.setVal(0._rt);
        b_mf.setVal(1._rt);

        int const idir = idim;
        int const lo_on = do_lo[idim];
        int const hi_on = do_hi[idim];
        if (!lo_on && !hi_on) { continue; }

        Real const pml_len = static_cast<Real>(ncell) * dx[idir];
        Real sigma_max = sigma_max_user;
        if (sigma_max < 0._rt && pml_len > 0._rt) {
            sigma_max = -(grade_m + 1._rt) * std::log(R) / (2._rt * eta0 * pml_len);
        }
        g_adi_cfs.sigma_max[idim] = sigma_max;

        for (MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
            auto const a_arr = a_mf.array(mfi);
            auto const b_arr = b_mf.array(mfi);
            auto const k_arr = kappa_mf.const_array(mfi);
            Box const bx = mfi.fabbox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const idx = (idir == 0) ? i : ((idir == 1) ? j : k);
                Real const coord =
                    problo[idir]
                    + (static_cast<Real>(idx - domain_lo[idir]) + 0.5_rt)
                        * dx[idir];
                Real const inner_lo = problo[idir] + pml_len;
                Real const inner_hi = probhi[idir] - pml_len;
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho <= 0._rt || pml_len <= 0._rt) {
                    a_arr(i,j,k) = 0._rt;
                    b_arr(i,j,k) = 1._rt;
                    return;
                }
                Real frac = rho / pml_len;
                frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                Real const poly = std::pow(frac, grade_m);
                Real const sig = sigma_max * poly;
                Real const alp = alpha_max * (1._rt - frac);
                Real const kapp = k_arr(i,j,k);
                Real const expo = -((sig / kapp) + alp) * dt_half / ep0;
                Real const bval = std::exp(expo);
                b_arr(i,j,k) = bval;
                Real const denom = kapp * (sig + kapp * alp);
                if (std::abs(denom) > 0._rt) {
                    a_arr(i,j,k) = (sig / denom) * (bval - 1._rt);
                } else {
                    a_arr(i,j,k) = 0._rt;
                }
            });
        }
        adi_pml_a[idim]->setBndry(0._rt);
        adi_pml_b[idim]->setBndry(1._rt);
        adi_pml_a[idim]->FillBoundary(Geom(0).periodicity());
        adi_pml_b[idim]->FillBoundary(Geom(0).periodicity());
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
            if (lo_on || hi_on) {
                Real const pml_len = static_cast<Real>(p.ncell) * p.dx[idir];
                Real const inner_lo = p.problo[idir] + pml_len;
                Real const inner_hi = p.probhi[idir] - pml_len;
                Real const rho = adi_pml_rho(
                    coord, inner_lo, inner_hi, lo_on, hi_on);
                if (rho > 0._rt && pml_len > 0._rt) {
                    Real frac = rho / pml_len;
                    frac = amrex::max(0._rt, amrex::min(1._rt, frac));
                    Real const poly = std::pow(frac, p.grade_m);
                    kappa = 1._rt + (p.kappa_max - 1._rt) * poly;
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
