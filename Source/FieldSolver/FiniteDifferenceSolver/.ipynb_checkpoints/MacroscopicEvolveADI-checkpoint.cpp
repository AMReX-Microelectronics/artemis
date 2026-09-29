#include "FiniteDifferenceSolver.H"
#include "AdiPML.H"

#include "MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <memory>

using namespace amrex;

namespace
{
    using FieldArray = std::array<std::unique_ptr<MultiFab>, 3>;
    using AdiFieldArray = std::array<FieldArray, 3>;

    struct AdiCoeffs
    {
        Real dx = 0._rt;
        Real dy = 0._rt;
        Real dz = 0._rt;
        Real inv_dx = 0._rt;
        Real inv_dy = 0._rt;
        Real inv_dz = 0._rt;
        Real dt = 0._rt;
        Real dtd2 = 0._rt;
        Real dtd4 = 0._rt;
    };

    struct AdiMaterialCoeffs
    {
        FieldArray Cb;
        FieldArray p;
        FieldArray kappa;
        FieldArray Db;
        FieldArray H;
    };

    struct PecConfig
    {
        std::array<bool, 3> normal = {false, false, false};
    };

    template <typename Arr>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real line_value (Arr const& arr, int solve_dir, int s,
                     int i, int j, int k) noexcept
    {
        if (solve_dir == 0) { return arr(s, j, k); }
        if (solve_dir == 1) { return arr(i, s, k); }
        return arr(i, j, s);
    }

    template <typename Arr>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_line_value (Arr const& arr, int solve_dir, int s,
                         int i, int j, int k, Real value) noexcept
    {
        if (solve_dir == 0) {
            arr(s, j, k) = value;
        } else if (solve_dir == 1) {
            arr(i, s, k) = value;
        } else {
            arr(i, j, s) = value;
        }
    }

    // Thomas tridiagonal solve. If rhs != nullptr, solve T x = rhs.
    // If rhs == nullptr, use the sparse Sherman-Morrison RHS u with u[0] = 1 and
    // u[n-1] = alpha/gamma (all other entries zero); alpha and gamma are only read then.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void solve_tridiagonal (Real const* a, Real const* b, Real const* c,
                            Real const* rhs, Real* x, Real* cprime, Real* dprime,
                            int n, Real alpha = 0._rt, Real gamma = 1._rt) noexcept
    {
        AMREX_ALWAYS_ASSERT(n >= 2);

        Real denom = b[0];
        AMREX_ALWAYS_ASSERT(std::abs(denom) > 0._rt);
        cprime[0] = c[0] / denom;
        dprime[0] = ((rhs != nullptr) ? rhs[0] : 1._rt) / denom;

        for (int i = 1; i < n; ++i) {
            denom = b[i] - a[i] * cprime[i - 1];
            AMREX_ALWAYS_ASSERT(std::abs(denom) > 0._rt);
            cprime[i] = (i < n - 1) ? c[i] / denom : 0._rt;
            Real const rhs_i = (rhs != nullptr) ? rhs[i]
                                                : ((i == n - 1) ? alpha / gamma : 0._rt);
            dprime[i] = (rhs_i - a[i] * dprime[i - 1]) / denom;
        }

        x[n - 1] = dprime[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            x[i] = dprime[i] - cprime[i] * x[i + 1];
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void solve_cyclic_tridiagonal (Real const* a, Real const* bb, Real const* c,
                                   Real alpha, Real beta, Real gamma,
                                   Real const* rhs, Real* x,
                                   Real* cprime, Real* dprime, Real* z, int n) noexcept
    {
        AMREX_ALWAYS_ASSERT(n > 2);
        AMREX_ALWAYS_ASSERT(std::abs(gamma) > 0._rt);

        // first solve Tx = rhs, with T = [a, bb, c]
        solve_tridiagonal(a, bb, c, rhs, x, cprime, dprime, n);

        // second solve Tz = u, u[0] = 1, u[n-1] = alpha/gamma
        solve_tridiagonal(a, bb, c, nullptr, z, cprime, dprime, n, alpha, gamma);

        Real const denom = 1._rt + gamma * z[0] + beta * z[n - 1];
        AMREX_ALWAYS_ASSERT(std::abs(denom) > 0._rt);
        Real const fact = (x[0] + beta * x[n - 1] / gamma) / denom;

        for (int i = 0; i < n; ++i) {
            x[i] -= fact * gamma * z[i];
        }
    }

    void solve_periodic_lines (
        MultiFab& field, MultiFab const& rhs,
        MultiFab const& Cb, MultiFab const& Db,
        int dir, Real inv_d2,
        MultiFab const* pec_mask,
        MultiFab const* stretch = nullptr)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field.ixType().nodeCentered(dir),
            "Macroscopic ADI periodic solve expects a nodal line along the implicit direction.");

        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(dir);
        int const hi = domain.bigEnd(dir);
        int const nsolve = hi - lo;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nsolve > 2,
            "ADI cyclic solve needs at least three unique cells per line.");

        field.ParallelCopy(rhs, 0, 0, 1);

        constexpr int n_line_work = 4;  // cprime, dprime, x, z
        constexpr int n_line_coeff = 3; // a, bb, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.validbox();
            auto const field_arr = field.array(mfi);
            auto const cb_arr = Cb.const_array(mfi);
            auto const db_arr = Db.const_array(mfi);
            Array4<Real const> pec_arr;
            if (pec_mask) {
                pec_arr = pec_mask->const_array(mfi);
            }
            Array4<Real const> k_arr;
            bool const use_k = stretch != nullptr;
            if (use_k) {
                k_arr = stretch->const_array(mfi);
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                bx.smallEnd(dir) == lo && bx.bigEnd(dir) == hi,
                "Each ADI pencil must span the full implicit direction.");

            Box const b2d = amrex::makeSlab(bx, dir, lo);
            Long const nlines = b2d.numPts();
            Gpu::AsyncVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::AsyncVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real* work = line_work.data();
            Real* coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real* cprime = work + line_id * nsolve * n_line_work;
                Real* dprime = cprime + nsolve;
                Real* x = dprime + nsolve;
                Real* z = x + nsolve;
                Real* a = coeff + line_id * nsolve * n_line_coeff;
                Real* bb = a + nsolve;
                Real* c = bb + nsolve;

                Real const db_seam = line_value(db_arr, dir, hi - 1, i, j, k);
                Real seam_scale = 1._rt;
                if (use_k) {
                    Real const k0 = line_value(k_arr, dir, lo, i, j, k);
                    Real const knm1 = line_value(k_arr, dir, hi - 1, i, j, k);
                    seam_scale = 1._rt / (k0 * knm1);
                }
                Real const alpha = -db_seam * seam_scale * inv_d2;
                Real const beta = -db_seam * seam_scale * inv_d2;

                // Physical coefficients first (needed for Sherman-Morrison corners).
                for (int p = 0; p < nsolve; ++p) {
                    int const s = lo + p;
                    Real const db_lo = (p == 0) ? db_seam
                                                : line_value(db_arr, dir, s - 1, i, j, k);
                    Real const db_hi = line_value(db_arr, dir, s, i, j, k);
                    Real k_center = 1._rt;
                    Real k_lo = 1._rt;
                    Real k_hi = 1._rt;
                    if (use_k) {
                        k_center = line_value(k_arr, dir, s, i, j, k);
                        k_lo = (p == 0)
                            ? line_value(k_arr, dir, hi - 1, i, j, k)
                            : line_value(k_arr, dir, s - 1, i, j, k);
                        k_hi = line_value(k_arr, dir, s, i, j, k);
                    }
                    Real const al = db_lo * inv_d2 / (k_center * k_lo);
                    Real const ga = db_hi * inv_d2 / (k_center * k_hi);
                    bb[p] = 1._rt / line_value(cb_arr, dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0._rt : -al;
                    c[p] = (p == nsolve - 1) ? 0._rt : -ga;
                    x[p] = line_value(field_arr, dir, lo + p, i, j, k);
                }

                Real const gamma = -bb[0];
                bb[0] -= gamma;
                bb[nsolve - 1] -= alpha * beta / gamma;

                // Blend PEC-mask nodes into identity rows with zero RHS.
                if (pec_mask) {
                    for (int p = 0; p < nsolve; ++p) {
                        Real const m = line_value(pec_arr, dir, lo + p, i, j, k);
                        a[p] *= m;
                        bb[p] = m * bb[p] + (1._rt - m);
                        c[p] *= m;
                        x[p] *= m;
                    }
                }

                solve_cyclic_tridiagonal(a, bb, c, alpha, beta, gamma, x, x,
                                         cprime, dprime, z, nsolve);

                if (pec_mask) {
                    for (int p = 0; p < nsolve; ++p) {
                        Real const m = line_value(pec_arr, dir, lo + p, i, j, k);
                        set_line_value(field_arr, dir, lo + p, i, j, k, m * x[p]);
                    }
                    Real const m0 = line_value(pec_arr, dir, lo, i, j, k);
                    set_line_value(field_arr, dir, hi, i, j, k, m0 * x[0]);
                } else {
                    for (int p = 0; p < nsolve; ++p) {
                        set_line_value(field_arr, dir, lo + p, i, j, k, x[p]);
                    }
                    set_line_value(field_arr, dir, hi, i, j, k, x[0]);
                }
            });
        }
    }

    void solve_dirichlet_nodal_lines (
        MultiFab& field, MultiFab const& rhs,
        MultiFab const& Cb, MultiFab const& Db,
        int dir, Real inv_d2,
        MultiFab const* pec_mask,
        MultiFab const* stretch = nullptr)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field.ixType().nodeCentered(dir),
            "Macroscopic ADI PEC solve expects a nodal line along the implicit direction.");

        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(dir);
        int const hi = domain.bigEnd(dir);
        int const nsolve = hi - lo - 1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nsolve >= 2,
            "ADI PEC solve needs at least two interior nodes per line.");

        field.ParallelCopy(rhs, 0, 0, 1);

        constexpr int n_line_work = 3;  // cprime, dprime, x
        constexpr int n_line_coeff = 3; // a, b, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.validbox();
            auto const field_arr = field.array(mfi);
            auto const cb_arr = Cb.const_array(mfi);
            auto const db_arr = Db.const_array(mfi);
            Array4<Real const> pec_arr;
            if (pec_mask != nullptr) {
                pec_arr = pec_mask->const_array(mfi);
            }
            Array4<Real const> k_arr;
            bool const use_k = stretch != nullptr;
            if (use_k) {
                k_arr = stretch->const_array(mfi);
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                bx.smallEnd(dir) == lo && bx.bigEnd(dir) == hi,
                "Each ADI PEC pencil must span the full implicit direction.");

            Box const b2d = amrex::makeSlab(bx, dir, lo + 1);
            Long const nlines = b2d.numPts();
            Gpu::AsyncVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::AsyncVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real* work = line_work.data();
            Real* coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real* cprime = work + line_id * nsolve * n_line_work;
                Real* dprime = cprime + nsolve;
                Real* x = dprime + nsolve;
                Real* a = coeff + line_id * nsolve * n_line_coeff;
                Real* b = a + nsolve;
                Real* c = b + nsolve;

                for (int p = 0; p < nsolve; ++p) {
                    int const s = lo + 1 + p;
                    Real const db_lo = line_value(db_arr, dir, s - 1, i, j, k);
                    Real const db_hi = line_value(db_arr, dir, s, i, j, k);
                    Real k_center = 1._rt;
                    Real k_lo = 1._rt;
                    Real k_hi = 1._rt;
                    if (use_k) {
                        k_center = line_value(k_arr, dir, s, i, j, k);
                        k_lo = line_value(k_arr, dir, s - 1, i, j, k);
                        k_hi = line_value(k_arr, dir, s, i, j, k);
                    }
                    Real const al = db_lo * inv_d2 / (k_center * k_lo);
                    Real const ga = db_hi * inv_d2 / (k_center * k_hi);
                    b[p] = 1._rt / line_value(cb_arr, dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0._rt : -al;
                    c[p] = (p == nsolve - 1) ? 0._rt : -ga;
                    x[p] = line_value(field_arr, dir, lo + 1 + p, i, j, k);
                    if (pec_mask) {
                        Real const m = line_value(pec_arr, dir, s, i, j, k);
                        a[p] *= m;
                        b[p] = m * b[p] + (1._rt - m);
                        c[p] *= m;
                        x[p] *= m;
                    }
                }

                solve_tridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                set_line_value(field_arr, dir, lo, i, j, k, 0._rt);
                if (pec_mask) {
                    for (int p = 0; p < nsolve; ++p) {
                        Real const m =
                            line_value(pec_arr, dir, lo + 1 + p, i, j, k);
                        set_line_value(
                            field_arr, dir, lo + 1 + p, i, j, k, m * x[p]);
                    }
                } else {
                    for (int p = 0; p < nsolve; ++p) {
                        set_line_value(
                            field_arr, dir, lo + 1 + p, i, j, k, x[p]);
                    }
                }
                set_line_value(field_arr, dir, hi, i, j, k, 0._rt);
            });
        }
    }

    PecConfig get_pec_config (Periodicity const& periodicity)
    {
        PecConfig pec;
        for (int dir = 0; dir < 3; ++dir) {
            bool const lo_pec = WarpX::field_boundary_lo[dir] == FieldBoundaryType::PEC;
            bool const hi_pec = WarpX::field_boundary_hi[dir] == FieldBoundaryType::PEC;
            bool const lo_pml = WarpX::field_boundary_lo[dir] == FieldBoundaryType::PML;
            bool const hi_pml = WarpX::field_boundary_hi[dir] == FieldBoundaryType::PML;
            bool const lo_dirichlet = lo_pec || lo_pml;
            bool const hi_dirichlet = hi_pec || hi_pml;
            if (lo_dirichlet || hi_dirichlet) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    lo_dirichlet && hi_dirichlet,
                    "Macroscopic ADI currently supports PEC/PML only on both domain walls "
                    "normal to one direction.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !periodicity.isPeriodic(dir),
                    "Macroscopic ADI PEC/PML direction must be non-periodic.");
                pec.normal[dir] = true;
            } else {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    periodicity.isPeriodic(dir),
                    "Macroscopic ADI currently supports periodic boundaries, with optional "
                    "PEC or in-domain PML walls normal to non-periodic directions.");
            }
        }
        return pec;
    }

    bool adi_pml_on ()
    {
        return WarpX::GetInstance().AdiPmlOn() != 0;
    }

    bool use_pec_dirichlet_solve (int e_comp, int solve_dir, PecConfig const& pec)
    {
        return pec.normal[solve_dir] && e_comp != solve_dir;
    }

    void solve_implicit_component (MultiFab& field, MultiFab const& rhs,
                                   MultiFab const& Cb, MultiFab const& Db,
                                   int e_comp, int solve_dir, Real inv_d2,
                                   PecConfig const& pec,
                                   MultiFab const* pec_mask,
                                   MultiFab const* stretch = nullptr)
    {
        if (use_pec_dirichlet_solve(e_comp, solve_dir, pec)) {
            solve_dirichlet_nodal_lines(field, rhs, Cb, Db, solve_dir, inv_d2,
                                        pec_mask, stretch);
        } else {
            solve_periodic_lines(field, rhs, Cb, Db, solve_dir, inv_d2,
                                 pec_mask, stretch);
        }
    }

    // Zero E on interior conductor edges marked by PEC_fp (mask value 0).
    void apply_pec_mask (FieldArray& Efield)
    {
        if (!WarpX::use_PEC_mask) { return; }
        WarpX& warpx = WarpX::GetInstance();
        for (int comp = 0; comp < 3; ++comp) {
            MultiFab* pec = warpx.get_pointer_PEC_fp(0, comp);
            if (pec == nullptr) { continue; }
            MultiFab::Multiply(*Efield[comp], *pec, 0, 0, 1, 0);
        }
    }

    void pin_pec_tangential_e (FieldArray& Efield, PecConfig const& pec)
    {
        for (int normal = 0; normal < 3; ++normal) {
            if (!pec.normal[normal]) { continue; }

            for (int comp = 0; comp < 3; ++comp) {
                if (comp == normal) { continue; }

                MultiFab& field = *Efield[comp];
                if (field.ixType().cellCentered(normal)) { continue; }

                Box const bounds = field.boxArray().minimalBox();
                int const lo = bounds.smallEnd(normal);
                int const hi = bounds.bigEnd(normal);

                for (MFIter mfi(field); mfi.isValid(); ++mfi) {
                    Box const& bx = mfi.validbox();
                    auto const arr = field.array(mfi);
                    if (lo >= bx.smallEnd(normal) && lo <= bx.bigEnd(normal)) {
                        Box const slab = amrex::makeSlab(bx, normal, lo);
                        ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            arr(i,j,k) = 0._rt;
                        });
                    }
                    if (hi >= bx.smallEnd(normal) && hi <= bx.bigEnd(normal)) {
                        Box const slab = amrex::makeSlab(bx, normal, hi);
                        ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            arr(i,j,k) = 0._rt;
                        });
                    }
                }
            }
        }
    }

    MultiFab make_rhs (MultiFab const& mf)
    {
        return MultiFab(mf.boxArray(), mf.DistributionMap(), 1, 0,
                        MFInfo().SetArena(The_Async_Arena()));
    }

    MultiFab make_coeff_like (MultiFab const& field, MultiFab const& coef)
    {
        BoxArray ba(field.boxArray());
        ba.convert(coef.ixType());
        return MultiFab(ba, field.DistributionMap(), 1, coef.nGrowVect(),
                        MFInfo().SetArena(The_Async_Arena()));
    }

    void copy_coeff_to_layout (
        MultiFab& dst, MultiFab const& src, Periodicity const& periodicity)
    {
        dst.ParallelCopy(src, 0, 0, 1, IntVect(0), dst.nGrowVect(), periodicity);
    }

    // Cell-centered CFS profiles onto a possibly staggered layout.
    // Initialize to `fill` first (1 for kappa/b, 0 for a). ParallelCopy then
    // fills overlapping indices; nodal planes that have no CC partner keep
    // `fill`. Do not index src Array4 at dst (i,j,k): that is OOB on hi nodes
    // and was leaving 1/kappa and the psi recursion with zeros/garbage.
    void copy_cc_profile_to_layout (
        MultiFab& dst, MultiFab const& src, Periodicity const& periodicity,
        Real fill)
    {
        dst.setVal(fill);
        copy_coeff_to_layout(dst, src, periodicity);
    }

    void add_soft_e_source_to_rhs (
        MultiFab& rhs, MultiFab const& Cb, int e_comp, int half_step)
    {
        if (WarpX::E_excitation_grid_s != "parse_e_excitation_grid_function") {
            return;
        }

        WarpX& warpx = WarpX::GetInstance();
        // (1/2) S^{n+1/4} on the first half, (1/2) S^{n+3/4} on the second.
        Real const time_frac = (half_step == 1) ? 0.25_rt : 0.75_rt;
        Real const weight = 0.5_rt;
        Real const time = warpx.gett_new(0) + time_frac * warpx.getdt(0);
        auto const field_parser =
            (e_comp == 0) ? warpx.Exfield_xt_grid_parser->compile<4>() :
            (e_comp == 1) ? warpx.Eyfield_xt_grid_parser->compile<4>() :
                            warpx.Ezfield_xt_grid_parser->compile<4>();
        auto const flag_parser =
            (e_comp == 0) ? warpx.Exfield_flag_parser->compile<3>() :
            (e_comp == 1) ? warpx.Eyfield_flag_parser->compile<3>() :
                            warpx.Ezfield_flag_parser->compile<3>();

        GpuArray<int, 3> rhs_stag;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            rhs_stag[idim] = rhs.ixType()[idim];
        }
        auto const problo = warpx.Geom(0).ProbLoArray();
        auto const dx = warpx.Geom(0).CellSizeArray();
        IntVect const rhs_nodal_flag = rhs.ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const rhs_arr = rhs.array(mfi);
            Array4<Real const> const cb_arr = Cb.const_array(mfi);
            Box const& bx = mfi.tilebox(rhs_nodal_flag);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, rhs_stag, problo, dx, x, y, z);
                Real const flag_type = flag_parser(x, y, z);
                if (flag_type > 2._rt) {
                    amrex::Abort("flag type for excitation must be <= 2");
                } else if (flag_type == 2._rt) {
                    // Flag=2 is a soft *field* increment (E += S), not an
                    // Ampère current. Divide by Cb so the contribution to E is
                    // independent of the material prefactor on the RHS curl terms.
                    rhs_arr(i,j,k) +=
                        weight * field_parser(x, y, z, time) / cb_arr(i,j,k);
                } else if (flag_type > 0._rt) {
                    amrex::Abort(
                        "Macroscopic ADI RHS-coupled E excitation supports only soft sources.");
                }
            });
        }
    }

    MultiFab make_copy (MultiFab const& mf)
    {
        MultiFab copy(mf.boxArray(), mf.DistributionMap(), 1, mf.nGrowVect(),
                      MFInfo().SetArena(The_Async_Arena()));
        MultiFab::Copy(copy, mf, 0, 0, 1, mf.nGrowVect());
        return copy;
    }

    std::unique_ptr<MultiFab> make_like (MultiFab const& mf)
    {
        return std::make_unique<MultiFab>(
            mf.boxArray(), mf.DistributionMap(), 1, mf.nGrowVect());
    }

    void define_material_coeffs (
        FieldArray const& Efield,
        FieldArray const& Bfield,
        AdiMaterialCoeffs& coeffs)
    {
        for (int comp = 0; comp < 3; ++comp) {
            coeffs.Cb[comp] = make_like(*Efield[comp]);
            coeffs.p[comp] = make_like(*Efield[comp]);
            coeffs.kappa[comp] = make_like(*Efield[comp]);
            coeffs.Db[comp] = make_like(*Bfield[comp]);
            coeffs.H[comp] = make_like(*Bfield[comp]);
        }
    }

    void fill_boundary_and_sync (FieldArray const& field,
                                 Periodicity const& periodicity)
    {
        for (auto const& component : field) {
            component->FillBoundaryAndSync(periodicity);
        }
    }

    void update_material_coeffs (
        AdiMaterialCoeffs& coeffs,
        FieldArray const& Bfield,
        Real const dt,
        Periodicity const& periodicity,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties)
    {
        MultiFab& sigma_mf = macroscopic_properties->getsigma_mf();
        MultiFab& epsilon_mf = macroscopic_properties->getepsilon_mf();
        MultiFab& mu_mf = macroscopic_properties->getmu_mf();

        amrex::GpuArray<int, 3> const& sigma_stag = macroscopic_properties->sigma_IndexType;
        amrex::GpuArray<int, 3> const& epsilon_stag = macroscopic_properties->epsilon_IndexType;
        amrex::GpuArray<int, 3> const& mu_stag = macroscopic_properties->mu_IndexType;
        amrex::GpuArray<int, 3> const& macro_cr = macroscopic_properties->macro_cr_ratio;

        std::array<amrex::GpuArray<int, 3> const*, 3> const e_stag = {
            &macroscopic_properties->Ex_IndexType,
            &macroscopic_properties->Ey_IndexType,
            &macroscopic_properties->Ez_IndexType};
        std::array<amrex::GpuArray<int, 3> const*, 3> const b_stag = {
            &macroscopic_properties->Bx_IndexType,
            &macroscopic_properties->By_IndexType,
            &macroscopic_properties->Bz_IndexType};

        WarpX& warpx = WarpX::GetInstance();
        bool const use_lumped_inductor = WarpX::use_lumped_inductor == 1;
        std::array<MultiFab const*, 3> inductance = {nullptr, nullptr, nullptr};
        if (use_lumped_inductor) {
            Inductor const& inductor = warpx.getInductor();
            inductance = {
                inductor.m_inductor_x_mf.get(),
                inductor.m_inductor_y_mf.get(),
                inductor.m_inductor_z_mf.get()};
        }
        auto const cell_size = warpx.Geom(0).CellSizeArray();
        std::array<Real, 3> const kappa_scale = {
            cell_size[0] / (cell_size[1] * cell_size[2]),
            cell_size[1] / (cell_size[0] * cell_size[2]),
            cell_size[2] / (cell_size[0] * cell_size[1])};

        for (int comp = 0; comp < 3; ++comp) {
            MultiFab& Cb = *coeffs.Cb[comp];
            MultiFab& p = *coeffs.p[comp];
            MultiFab& kappa_mf = *coeffs.kappa[comp];
            MultiFab const* inductance_mf = inductance[comp];

            // Fill only the valid region, then exchange ghosts below. Including the
            // grown tilebox here makes sample::Interp read sigma/epsilon outside their
            // fabs (staggering offset), which segfaults under OpenMP.
            for (MFIter mfi(Cb, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real> const cb_arr = Cb.array(mfi);
                Array4<Real> const p_arr = p.array(mfi);
                Array4<Real> const kappa_arr = kappa_mf.array(mfi);
                Array4<Real> const sigma_arr = sigma_mf.array(mfi);
                Array4<Real> const eps_arr = epsilon_mf.array(mfi);
                Array4<Real const> inductance_arr;
                if (inductance_mf != nullptr) {
                    inductance_arr = inductance_mf->const_array(mfi);
                }
                Box const& bx = mfi.tilebox(Cb.ixType().toIntVect());
                auto const& estag = *e_stag[comp];
                Real const scale = kappa_scale[comp];
                bool const has_inductor = inductance_mf != nullptr;
                int const scomp = 0;

                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const sigma = ablastr::coarsen::sample::Interp(
                        sigma_arr, sigma_stag, estag, macro_cr, i, j, k, scomp);
                    Real const eps = ablastr::coarsen::sample::Interp(
                        eps_arr, epsilon_stag, estag, macro_cr, i, j, k, scomp);
                    Real const L = has_inductor ? inductance_arr(i,j,k) : 0._rt;
                    Real const kappa = (L > 0._rt) ? scale / L : 0._rt;
                    // Centering J_L in Ampere and E in dJ_L/dt over a half-step
                    // adds dt*kappa/8 to both electric endpoint coefficients.
                    Real const dt2_kappa_d4 = 0.25_rt * dt * dt * kappa;
                    Real const denom = 4._rt * eps + sigma * dt + dt2_kappa_d4;
                    Real const Cb_val = 2._rt * dt / denom;
                    cb_arr(i,j,k) = Cb_val;
                    p_arr(i,j,k) =
                        (4._rt * eps - sigma * dt - dt2_kappa_d4) / (2._rt * dt);
                    kappa_arr(i,j,k) = kappa;
                });
            }

            MultiFab& Db = *coeffs.Db[comp];
            MultiFab& H = *coeffs.H[comp];
            MultiFab const& B = *Bfield[comp];

            for (MFIter mfi(Db, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real> const db_arr = Db.array(mfi);
                Array4<Real> const h_arr = H.array(mfi);
                Array4<Real const> const b_arr = B.const_array(mfi);
                Array4<Real> const mu_arr = mu_mf.array(mfi);
                Box const& bx = mfi.tilebox(Db.ixType().toIntVect());
                auto const& bstag = *b_stag[comp];
                int const scomp = 0;

                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const mu = ablastr::coarsen::sample::Interp(
                        mu_arr, mu_stag, bstag, macro_cr, i, j, k, scomp);
                    db_arr(i,j,k) = dt / (2._rt * mu);
                    h_arr(i,j,k) = b_arr(i,j,k) / mu;
                });
            }
        }

        fill_boundary_and_sync(coeffs.Cb, periodicity);
        fill_boundary_and_sync(coeffs.p, periodicity);
        fill_boundary_and_sync(coeffs.kappa, periodicity);
        fill_boundary_and_sync(coeffs.Db, periodicity);
        fill_boundary_and_sync(coeffs.H, periodicity);
    }

    void copy_fields (FieldArray& dst, FieldArray const& src,
                      Periodicity const& periodicity)
    {
        for (int component = 0; component < 3; ++component) {
            dst[component]->ParallelCopy(
                *src[component], 0, 0, 1, IntVect(0),
                dst[component]->nGrowVect(), periodicity);
        }
    }

    void copy_field_component (FieldArray& dst, FieldArray const& src,
                               int component, Periodicity const& periodicity)
    {
        dst[component]->ParallelCopy(
            *src[component], 0, 0, 1, IntVect(0), IntVect(0), periodicity);
    }

    void add_lumped_inductor_current_to_rhs (
        MultiFab& rhs, MultiFab const& kappa,
        int e_comp, Periodicity const& periodicity)
    {
        if (WarpX::use_lumped_inductor != 1) { return; }

        WarpX& warpx = WarpX::GetInstance();
        MultiFab const& current = *warpx.get_pointer_current_fp(0, e_comp);
        MultiFab current_field = make_rhs(rhs);
        MultiFab kappa_field = make_rhs(rhs);
        copy_coeff_to_layout(current_field, current, periodicity);
        copy_coeff_to_layout(kappa_field, kappa, periodicity);

        for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const rhs_arr = rhs.array(mfi);
            Array4<Real const> const current_arr = current_field.const_array(mfi);
            Array4<Real const> const kappa_arr = kappa_field.const_array(mfi);
            Box const& bx = mfi.tilebox(rhs.ixType().toIntVect());
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (kappa_arr(i,j,k) > 0._rt) {
                    // The tridiagonal equations are normalized by Cb^L.
                    rhs_arr(i,j,k) -= current_arr(i,j,k);
                }
            });
        }
    }

    // First half-step RHS: implicit Ex along y.
    MultiFab build_rhs_ex1 (
        MultiFab const& ex, MultiFab const& ey,
        MultiFab const& hy, MultiFab const& hz,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ex);
        MultiFab p_field = make_rhs(ex);
        MultiFab cb_field = make_rhs(ex);
        MultiFab db_field = make_coeff_like(ex, *mat.Db[2]);
        copy_coeff_to_layout(p_field, *mat.p[0], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[0], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[2], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const hy_arr = hy.const_array(mfi);
            auto const hz_arr = hz.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& bx = mfi.validbox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i, j-1, k);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hz_arr(i,j,k) - hz_arr(i,j-1,k)) * c.inv_dy -
                    (hy_arr(i,j,k) - hy_arr(i,j,k-1)) * c.inv_dz;
                Real const ey_lo = ey_arr(i+1,j-1,k) - ey_arr(i,j-1,k);
                Real const ey_hi = ey_arr(i+1,j,k) - ey_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                    + q * c.inv_dx * c.inv_dy * ey_lo
                    - r * c.inv_dx * c.inv_dy * ey_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab ky_field = make_rhs(ex);
            MultiFab kz_field = make_rhs(ex);
            MultiFab kx_db = make_coeff_like(ex, *mat.Db[2]);
            MultiFab psi_e0 = make_rhs(ex);
            MultiFab psi_e1 = make_rhs(ex);
            MultiFab psi_h0 = make_coeff_like(ex, *mat.Db[2]);
            MultiFab psi_h1 = make_coeff_like(ex, *mat.Db[2]);
            copy_cc_profile_to_layout(ky_field, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_cc_profile_to_layout(kz_field, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_cc_profile_to_layout(kx_db, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EXY), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EXZ), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HZY), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HZX), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const ky_arr = ky_field.const_array(mfi);
                auto const kz_arr = kz_field.const_array(mfi);
                auto const kx_arr = kx_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& bx = mfi.validbox();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j-1, k);
                    Real const r = db_arr(i, j, k);
                    Real const ky = ky_arr(i,j,k);
                    Real const kz = kz_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dy / ky) * (hz_arr(i,j,k) - hz_arr(i,j-1,k)) -
                        (c.inv_dz / kz) * (hy_arr(i,j,k) - hy_arr(i,j,k-1));
                    Real const ey_lo = ey_arr(i+1,j-1,k) - ey_arr(i,j-1,k);
                    Real const ey_hi = ey_arr(i+1,j,k) - ey_arr(i,j,k);
                    Real const kx_lo = kx_arr(i, j-1, k);
                    Real const kx_hi = kx_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i,j-1,k) - ph1(i,j-1,k);
                    Real const psi_h_term =
                        -(r * c.inv_dy / ky) * psi_h_hi +
                         (q * c.inv_dy / ky) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                        + q * c.inv_dx * c.inv_dy * ey_lo / (ky * kx_lo)
                        - r * c.inv_dx * c.inv_dy * ey_hi / (ky * kx_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[0], 0, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 0, 1);
        return rhs;
    }

    // First half-step RHS: implicit Ey along z.
    MultiFab build_rhs_ey1 (
        MultiFab const& ey, MultiFab const& ez,
        MultiFab const& hx, MultiFab const& hz,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ey);
        MultiFab p_field = make_rhs(ey);
        MultiFab cb_field = make_rhs(ey);
        MultiFab db_field = make_coeff_like(ey, *mat.Db[0]);
        copy_coeff_to_layout(p_field, *mat.p[1], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[1], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[0], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const hx_arr = hx.const_array(mfi);
            auto const hz_arr = hz.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i, j, k-1);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hx_arr(i,j,k) - hx_arr(i,j,k-1)) * c.inv_dz -
                    (hz_arr(i,j,k) - hz_arr(i-1,j,k)) * c.inv_dx;
                Real const ez_lo = ez_arr(i,j+1,k-1) - ez_arr(i,j,k-1);
                Real const ez_hi = ez_arr(i,j+1,k) - ez_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                    + q * c.inv_dy * c.inv_dz * ez_lo
                    - r * c.inv_dy * c.inv_dz * ez_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab kz_field = make_rhs(ey);
            MultiFab kx_field = make_rhs(ey);
            MultiFab ky_db = make_coeff_like(ey, *mat.Db[0]);
            MultiFab psi_e0 = make_rhs(ey);
            MultiFab psi_e1 = make_rhs(ey);
            MultiFab psi_h0 = make_coeff_like(ey, *mat.Db[0]);
            MultiFab psi_h1 = make_coeff_like(ey, *mat.Db[0]);
            copy_cc_profile_to_layout(kz_field, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_cc_profile_to_layout(kx_field, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_cc_profile_to_layout(ky_db, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EYZ), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EYX), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HXZ), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HXY), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const kz_arr = kz_field.const_array(mfi);
                auto const kx_arr = kx_field.const_array(mfi);
                auto const ky_arr = ky_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j, k-1);
                    Real const r = db_arr(i, j, k);
                    Real const kz = kz_arr(i,j,k);
                    Real const kx = kx_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dz / kz) * (hx_arr(i,j,k) - hx_arr(i,j,k-1)) -
                        (c.inv_dx / kx) * (hz_arr(i,j,k) - hz_arr(i-1,j,k));
                    Real const ez_lo = ez_arr(i,j+1,k-1) - ez_arr(i,j,k-1);
                    Real const ez_hi = ez_arr(i,j+1,k) - ez_arr(i,j,k);
                    Real const ky_lo = ky_arr(i, j, k-1);
                    Real const ky_hi = ky_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i,j,k-1) - ph1(i,j,k-1);
                    Real const psi_h_term =
                        -(r * c.inv_dz / kz) * psi_h_hi +
                         (q * c.inv_dz / kz) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                        + q * c.inv_dy * c.inv_dz * ez_lo / (kz * ky_lo)
                        - r * c.inv_dy * c.inv_dz * ez_hi / (kz * ky_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[1], 1, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 1, 1);
        return rhs;
    }

    // First half-step RHS: implicit Ez along x.
    MultiFab build_rhs_ez1 (
        MultiFab const& ez, MultiFab const& ex,
        MultiFab const& hx, MultiFab const& hy,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ez);
        MultiFab p_field = make_rhs(ez);
        MultiFab cb_field = make_rhs(ez);
        MultiFab db_field = make_coeff_like(ez, *mat.Db[1]);
        copy_coeff_to_layout(p_field, *mat.p[2], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[2], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[1], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const hx_arr = hx.const_array(mfi);
            auto const hy_arr = hy.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i-1, j, k);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hy_arr(i,j,k) - hy_arr(i-1,j,k)) * c.inv_dx -
                    (hx_arr(i,j,k) - hx_arr(i,j-1,k)) * c.inv_dy;
                Real const ex_lo = ex_arr(i-1,j,k+1) - ex_arr(i-1,j,k);
                Real const ex_hi = ex_arr(i,j,k+1) - ex_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                    + q * c.inv_dz * c.inv_dx * ex_lo
                    - r * c.inv_dz * c.inv_dx * ex_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab kx_field = make_rhs(ez);
            MultiFab ky_field = make_rhs(ez);
            MultiFab kz_db = make_coeff_like(ez, *mat.Db[1]);
            MultiFab psi_e0 = make_rhs(ez);
            MultiFab psi_e1 = make_rhs(ez);
            MultiFab psi_h0 = make_coeff_like(ez, *mat.Db[1]);
            MultiFab psi_h1 = make_coeff_like(ez, *mat.Db[1]);
            copy_cc_profile_to_layout(kx_field, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_cc_profile_to_layout(ky_field, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_cc_profile_to_layout(kz_db, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EZX), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EZY), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HYX), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HYZ), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const kx_arr = kx_field.const_array(mfi);
                auto const ky_arr = ky_field.const_array(mfi);
                auto const kz_arr = kz_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i-1, j, k);
                    Real const r = db_arr(i, j, k);
                    Real const kx = kx_arr(i,j,k);
                    Real const ky = ky_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dx / kx) * (hy_arr(i,j,k) - hy_arr(i-1,j,k)) -
                        (c.inv_dy / ky) * (hx_arr(i,j,k) - hx_arr(i,j-1,k));
                    Real const ex_lo = ex_arr(i-1,j,k+1) - ex_arr(i-1,j,k);
                    Real const ex_hi = ex_arr(i,j,k+1) - ex_arr(i,j,k);
                    Real const kz_lo = kz_arr(i-1, j, k);
                    Real const kz_hi = kz_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i-1,j,k) - ph1(i-1,j,k);
                    Real const psi_h_term =
                        -(r * c.inv_dx / kx) * psi_h_hi +
                         (q * c.inv_dx / kx) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                        + q * c.inv_dz * c.inv_dx * ex_lo / (kx * kz_lo)
                        - r * c.inv_dz * c.inv_dx * ex_hi / (kx * kz_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[2], 2, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 2, 1);
        return rhs;
    }

    // Second half-step RHS: implicit Ex along z.
    MultiFab build_rhs_ex2 (
        MultiFab const& ex, MultiFab const& ez,
        MultiFab const& hy, MultiFab const& hz,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ex);
        MultiFab p_field = make_rhs(ex);
        MultiFab cb_field = make_rhs(ex);
        MultiFab db_field = make_coeff_like(ex, *mat.Db[1]);
        copy_coeff_to_layout(p_field, *mat.p[0], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[0], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[1], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const hy_arr = hy.const_array(mfi);
            auto const hz_arr = hz.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i, j, k-1);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hz_arr(i,j,k) - hz_arr(i,j-1,k)) * c.inv_dy -
                    (hy_arr(i,j,k) - hy_arr(i,j,k-1)) * c.inv_dz;
                Real const ez_lo = ez_arr(i+1,j,k-1) - ez_arr(i,j,k-1);
                Real const ez_hi = ez_arr(i+1,j,k) - ez_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                    + q * c.inv_dx * c.inv_dz * ez_lo
                    - r * c.inv_dx * c.inv_dz * ez_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab ky_field = make_rhs(ex);
            MultiFab kz_field = make_rhs(ex);
            MultiFab kx_db = make_coeff_like(ex, *mat.Db[1]);
            MultiFab psi_e0 = make_rhs(ex);
            MultiFab psi_e1 = make_rhs(ex);
            MultiFab psi_h0 = make_coeff_like(ex, *mat.Db[1]);
            MultiFab psi_h1 = make_coeff_like(ex, *mat.Db[1]);
            copy_cc_profile_to_layout(ky_field, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_cc_profile_to_layout(kz_field, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_cc_profile_to_layout(kx_db, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EXY), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EXZ), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HYX), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HYZ), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const ky_arr = ky_field.const_array(mfi);
                auto const kz_arr = kz_field.const_array(mfi);
                auto const kx_arr = kx_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j, k-1);
                    Real const r = db_arr(i, j, k);
                    Real const ky = ky_arr(i,j,k);
                    Real const kz = kz_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dy / ky) * (hz_arr(i,j,k) - hz_arr(i,j-1,k)) -
                        (c.inv_dz / kz) * (hy_arr(i,j,k) - hy_arr(i,j,k-1));
                    Real const ez_lo = ez_arr(i+1,j,k-1) - ez_arr(i,j,k-1);
                    Real const ez_hi = ez_arr(i+1,j,k) - ez_arr(i,j,k);
                    Real const kx_lo = kx_arr(i, j, k-1);
                    Real const kx_hi = kx_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i,j,k-1) - ph1(i,j,k-1);
                    Real const psi_h_term =
                        -(r * c.inv_dz / kz) * psi_h_hi +
                         (q * c.inv_dz / kz) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                        + q * c.inv_dx * c.inv_dz * ez_lo / (kz * kx_lo)
                        - r * c.inv_dx * c.inv_dz * ez_hi / (kz * kx_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[0], 0, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 0, 2);
        return rhs;
    }

    // Second half-step RHS: implicit Ey along x.
    MultiFab build_rhs_ey2 (
        MultiFab const& ey, MultiFab const& ex,
        MultiFab const& hx, MultiFab const& hz,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ey);
        MultiFab p_field = make_rhs(ey);
        MultiFab cb_field = make_rhs(ey);
        MultiFab db_field = make_coeff_like(ey, *mat.Db[2]);
        copy_coeff_to_layout(p_field, *mat.p[1], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[1], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[2], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const hx_arr = hx.const_array(mfi);
            auto const hz_arr = hz.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i-1, j, k);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hx_arr(i,j,k) - hx_arr(i,j,k-1)) * c.inv_dz -
                    (hz_arr(i,j,k) - hz_arr(i-1,j,k)) * c.inv_dx;
                Real const ex_lo = ex_arr(i-1,j+1,k) - ex_arr(i-1,j,k);
                Real const ex_hi = ex_arr(i,j+1,k) - ex_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                    + q * c.inv_dy * c.inv_dx * ex_lo
                    - r * c.inv_dy * c.inv_dx * ex_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab kz_field = make_rhs(ey);
            MultiFab kx_field = make_rhs(ey);
            MultiFab ky_db = make_coeff_like(ey, *mat.Db[2]);
            MultiFab psi_e0 = make_rhs(ey);
            MultiFab psi_e1 = make_rhs(ey);
            MultiFab psi_h0 = make_coeff_like(ey, *mat.Db[2]);
            MultiFab psi_h1 = make_coeff_like(ey, *mat.Db[2]);
            copy_cc_profile_to_layout(kz_field, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_cc_profile_to_layout(kx_field, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_cc_profile_to_layout(ky_db, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EYZ), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EYX), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HZY), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HZX), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const kz_arr = kz_field.const_array(mfi);
                auto const kx_arr = kx_field.const_array(mfi);
                auto const ky_arr = ky_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i-1, j, k);
                    Real const r = db_arr(i, j, k);
                    Real const kz = kz_arr(i,j,k);
                    Real const kx = kx_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dz / kz) * (hx_arr(i,j,k) - hx_arr(i,j,k-1)) -
                        (c.inv_dx / kx) * (hz_arr(i,j,k) - hz_arr(i-1,j,k));
                    Real const ex_lo = ex_arr(i-1,j+1,k) - ex_arr(i-1,j,k);
                    Real const ex_hi = ex_arr(i,j+1,k) - ex_arr(i,j,k);
                    Real const ky_lo = ky_arr(i-1, j, k);
                    Real const ky_hi = ky_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i-1,j,k) - ph1(i-1,j,k);
                    Real const psi_h_term =
                        -(r * c.inv_dx / kx) * psi_h_hi +
                         (q * c.inv_dx / kx) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                        + q * c.inv_dy * c.inv_dx * ex_lo / (kx * ky_lo)
                        - r * c.inv_dy * c.inv_dx * ex_hi / (kx * ky_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[1], 1, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 1, 2);
        return rhs;
    }

    // Second half-step RHS: implicit Ez along y.
    MultiFab build_rhs_ez2 (
        MultiFab const& ez, MultiFab const& ey,
        MultiFab const& hx, MultiFab const& hy,
        AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity)
    {
        MultiFab rhs = make_rhs(ez);
        MultiFab p_field = make_rhs(ez);
        MultiFab cb_field = make_rhs(ez);
        MultiFab db_field = make_coeff_like(ez, *mat.Db[0]);
        copy_coeff_to_layout(p_field, *mat.p[2], periodicity);
        copy_coeff_to_layout(cb_field, *mat.Cb[2], periodicity);
        copy_coeff_to_layout(db_field, *mat.Db[0], periodicity);
        if (!adi_pml_on()) {
        for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
            auto const rhs_arr = rhs.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const hx_arr = hx.const_array(mfi);
            auto const hy_arr = hy.const_array(mfi);
            auto const p_arr = p_field.const_array(mfi);
            auto const db_arr = db_field.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const q = db_arr(i, j-1, k);
                Real const r = db_arr(i, j, k);
                Real const curl_h =
                    (hy_arr(i,j,k) - hy_arr(i-1,j,k)) * c.inv_dx -
                    (hx_arr(i,j,k) - hx_arr(i,j-1,k)) * c.inv_dy;
                Real const ey_lo = ey_arr(i,j-1,k+1) - ey_arr(i,j-1,k);
                Real const ey_hi = ey_arr(i,j,k+1) - ey_arr(i,j,k);
                rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                    + q * c.inv_dz * c.inv_dy * ey_lo
                    - r * c.inv_dz * c.inv_dy * ey_hi;
            });
        }
        } else {
            WarpX& warpx = WarpX::GetInstance();
            MultiFab kx_field = make_rhs(ez);
            MultiFab ky_field = make_rhs(ez);
            MultiFab kz_db = make_coeff_like(ez, *mat.Db[0]);
            MultiFab psi_e0 = make_rhs(ez);
            MultiFab psi_e1 = make_rhs(ez);
            MultiFab psi_h0 = make_coeff_like(ez, *mat.Db[0]);
            MultiFab psi_h1 = make_coeff_like(ez, *mat.Db[0]);
            copy_cc_profile_to_layout(kx_field, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
            copy_cc_profile_to_layout(ky_field, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
            copy_cc_profile_to_layout(kz_db, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
            copy_coeff_to_layout(psi_e0, warpx.get_adi_psi_e(AdiPsiE::EZX), periodicity);
            copy_coeff_to_layout(psi_e1, warpx.get_adi_psi_e(AdiPsiE::EZY), periodicity);
            copy_coeff_to_layout(psi_h0, warpx.get_adi_psi_h(AdiPsiH::HXZ), periodicity);
            copy_coeff_to_layout(psi_h1, warpx.get_adi_psi_h(AdiPsiH::HXY), periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const kx_arr = kx_field.const_array(mfi);
                auto const ky_arr = ky_field.const_array(mfi);
                auto const kz_arr = kz_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const ph0 = psi_h0.const_array(mfi);
                auto const ph1 = psi_h1.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j-1, k);
                    Real const r = db_arr(i, j, k);
                    Real const kx = kx_arr(i,j,k);
                    Real const ky = ky_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dx / kx) * (hy_arr(i,j,k) - hy_arr(i-1,j,k)) -
                        (c.inv_dy / ky) * (hx_arr(i,j,k) - hx_arr(i,j-1,k));
                    Real const ey_lo = ey_arr(i,j-1,k+1) - ey_arr(i,j-1,k);
                    Real const ey_hi = ey_arr(i,j,k+1) - ey_arr(i,j,k);
                    Real const kz_lo = kz_arr(i, j-1, k);
                    Real const kz_hi = kz_arr(i, j, k);
                    Real const psi_h_hi = ph0(i,j,k) - ph1(i,j,k);
                    Real const psi_h_lo = ph0(i,j-1,k) - ph1(i,j-1,k);
                    Real const psi_h_term =
                        -(r * c.inv_dy / ky) * psi_h_hi +
                         (q * c.inv_dy / ky) * psi_h_lo;
                    Real const psi_e_term = pe1(i,j,k) - pe0(i,j,k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                        + q * c.inv_dz * c.inv_dy * ey_lo / (ky * kz_lo)
                        - r * c.inv_dz * c.inv_dy * ey_hi / (ky * kz_hi)
                        + psi_h_term + psi_e_term;
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[2], 2, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 2, 2);
        return rhs;
    }

    void solve_implicit_ex1 (MultiFab& ex, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ex);
        MultiFab Db = make_coeff_like(ex, *mat.Db[2]);
        copy_coeff_to_layout(Cb, *mat.Cb[0], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[2], periodicity);
        MultiFab const* pec_mask = pec_masks[1][0].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ex);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(1), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ex, rhs, Cb, Db,
                                 0, 1, c.inv_dy * c.inv_dy, pec, pec_mask, stretch);
    }

    void solve_implicit_ey1 (MultiFab& ey, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ey);
        MultiFab Db = make_coeff_like(ey, *mat.Db[0]);
        copy_coeff_to_layout(Cb, *mat.Cb[1], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[0], periodicity);
        MultiFab const* pec_mask = pec_masks[2][1].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ey);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(2), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ey, rhs, Cb, Db,
                                 1, 2, c.inv_dz * c.inv_dz, pec, pec_mask, stretch);
    }

    void solve_implicit_ez1 (MultiFab& ez, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ez);
        MultiFab Db = make_coeff_like(ez, *mat.Db[1]);
        copy_coeff_to_layout(Cb, *mat.Cb[2], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[1], periodicity);
        MultiFab const* pec_mask = pec_masks[0][2].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ez);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(0), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ez, rhs, Cb, Db,
                                 2, 0, c.inv_dx * c.inv_dx, pec, pec_mask, stretch);
    }

    void solve_implicit_ex2 (MultiFab& ex, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ex);
        MultiFab Db = make_coeff_like(ex, *mat.Db[1]);
        copy_coeff_to_layout(Cb, *mat.Cb[0], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[1], periodicity);
        MultiFab const* pec_mask = pec_masks[2][0].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ex);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(2), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ex, rhs, Cb, Db,
                                 0, 2, c.inv_dz * c.inv_dz, pec, pec_mask, stretch);
    }

    void solve_implicit_ey2 (MultiFab& ey, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ey);
        MultiFab Db = make_coeff_like(ey, *mat.Db[2]);
        copy_coeff_to_layout(Cb, *mat.Cb[1], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[2], periodicity);
        MultiFab const* pec_mask = pec_masks[0][1].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ey);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(0), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ey, rhs, Cb, Db,
                                 1, 0, c.inv_dx * c.inv_dx, pec, pec_mask, stretch);
    }

    void solve_implicit_ez2 (MultiFab& ez, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab Cb = make_rhs(ez);
        MultiFab Db = make_coeff_like(ez, *mat.Db[0]);
        copy_coeff_to_layout(Cb, *mat.Cb[2], periodicity);
        copy_coeff_to_layout(Db, *mat.Db[0], periodicity);
        MultiFab const* pec_mask = pec_masks[1][2].get();
        MultiFab stretch_mf;
        MultiFab const* stretch = nullptr;
        if (adi_pml_on()) {
            stretch_mf = make_rhs(ez);
            copy_cc_profile_to_layout(stretch_mf, WarpX::GetInstance().get_adi_pml_stretch(1), periodicity, 1._rt);
            stretch = &stretch_mf;
        }
        solve_implicit_component(ez, rhs, Cb, Db,
                                 2, 1, c.inv_dy * c.inv_dy, pec, pec_mask, stretch);
    }

    void step_jx (MultiFab& jx, MultiFab const& ex_old, MultiFab const& ex_new,
                  MultiFab const& kappa_x, AdiCoeffs const& c)
    {
        for (MFIter mfi(jx); mfi.isValid(); ++mfi) {
            auto const jx_arr = jx.array(mfi);
            auto const ex_old_arr = ex_old.const_array(mfi);
            auto const ex_new_arr = ex_new.const_array(mfi);
            auto const kappa_x_arr = kappa_x.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                jx_arr(i,j,k) += c.dtd4 * kappa_x_arr(i,j,k)
                    * (ex_new_arr(i,j,k) + ex_old_arr(i,j,k));
            });
        }
    }

    void step_jy (MultiFab& jy, MultiFab const& ey_old, MultiFab const& ey_new,
                  MultiFab const& kappa_y, AdiCoeffs const& c)
    {
        for (MFIter mfi(jy); mfi.isValid(); ++mfi) {
            auto const jy_arr = jy.array(mfi);
            auto const ey_old_arr = ey_old.const_array(mfi);
            auto const ey_new_arr = ey_new.const_array(mfi);
            auto const kappa_y_arr = kappa_y.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                jy_arr(i,j,k) += c.dtd4 * kappa_y_arr(i,j,k)
                    * (ey_new_arr(i,j,k) + ey_old_arr(i,j,k));
            });
        }
    }

    void step_jz (MultiFab& jz, MultiFab const& ez_old, MultiFab const& ez_new,
                  MultiFab const& kappa_z, AdiCoeffs const& c)
    {
        for (MFIter mfi(jz); mfi.isValid(); ++mfi) {
            auto const jz_arr = jz.array(mfi);
            auto const ez_old_arr = ez_old.const_array(mfi);
            auto const ez_new_arr = ez_new.const_array(mfi);
            auto const kappa_z_arr = kappa_z.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                jz_arr(i,j,k) += c.dtd4 * kappa_z_arr(i,j,k)
                    * (ez_new_arr(i,j,k) + ez_old_arr(i,j,k));
            });
        }
    }

    void step_bx (MultiFab& bx, MultiFab const& ey, MultiFab const& ez, AdiCoeffs const& c)
    {
        for (MFIter mfi(bx); mfi.isValid(); ++mfi) {
            auto const bx_arr = bx.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bx_arr(i,j,k) += c.dtd2 * ((ey_arr(i,j,k+1) - ey_arr(i,j,k)) * c.inv_dz -
                                           (ez_arr(i,j+1,k) - ez_arr(i,j,k)) * c.inv_dy);
            });
        }
    }

    void step_by (MultiFab& by, MultiFab const& ez, MultiFab const& ex, AdiCoeffs const& c)
    {
        for (MFIter mfi(by); mfi.isValid(); ++mfi) {
            auto const by_arr = by.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                by_arr(i,j,k) += c.dtd2 * ((ez_arr(i+1,j,k) - ez_arr(i,j,k)) * c.inv_dx -
                                           (ex_arr(i,j,k+1) - ex_arr(i,j,k)) * c.inv_dz);
            });
        }
    }

    void step_bz (MultiFab& bz, MultiFab const& ex, MultiFab const& ey, AdiCoeffs const& c)
    {
        for (MFIter mfi(bz); mfi.isValid(); ++mfi) {
            auto const bz_arr = bz.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bz_arr(i,j,k) += c.dtd2 * ((ex_arr(i,j+1,k) - ex_arr(i,j,k)) * c.inv_dy -
                                           (ey_arr(i+1,j,k) - ey_arr(i,j,k)) * c.inv_dx);
            });
        }
    }

    void step_bx_pml (MultiFab& bx, MultiFab const& ey, MultiFab const& ez,
                      AdiCoeffs const& c, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        MultiFab kz = make_rhs(bx);
        MultiFab ky = make_rhs(bx);
        copy_cc_profile_to_layout(kz, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
        copy_cc_profile_to_layout(ky, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
        MultiFab const& psi_hxz = warpx.get_adi_psi_h(AdiPsiH::HXZ);
        MultiFab const& psi_hxy = warpx.get_adi_psi_h(AdiPsiH::HXY);
        for (MFIter mfi(bx); mfi.isValid(); ++mfi) {
            auto const bx_arr = bx.array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const kz_arr = kz.const_array(mfi);
            auto const ky_arr = ky.const_array(mfi);
            auto const phxz = psi_hxz.const_array(mfi);
            auto const phxy = psi_hxy.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bx_arr(i,j,k) += c.dtd2 * (
                    (c.inv_dz / kz_arr(i,j,k)) * (ey_arr(i,j,k+1) - ey_arr(i,j,k)) -
                    (c.inv_dy / ky_arr(i,j,k)) * (ez_arr(i,j+1,k) - ez_arr(i,j,k)) +
                    phxz(i,j,k) - phxy(i,j,k));
            });
        }
    }

    void step_by_pml (MultiFab& by, MultiFab const& ez, MultiFab const& ex,
                      AdiCoeffs const& c, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        MultiFab kx = make_rhs(by);
        MultiFab kz = make_rhs(by);
        copy_cc_profile_to_layout(kx, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
        copy_cc_profile_to_layout(kz, warpx.get_adi_pml_stretch(2), periodicity, 1._rt);
        MultiFab const& psi_hyx = warpx.get_adi_psi_h(AdiPsiH::HYX);
        MultiFab const& psi_hyz = warpx.get_adi_psi_h(AdiPsiH::HYZ);
        for (MFIter mfi(by); mfi.isValid(); ++mfi) {
            auto const by_arr = by.array(mfi);
            auto const ez_arr = ez.const_array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const kx_arr = kx.const_array(mfi);
            auto const kz_arr = kz.const_array(mfi);
            auto const phyx = psi_hyx.const_array(mfi);
            auto const phyz = psi_hyz.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                by_arr(i,j,k) += c.dtd2 * (
                    (c.inv_dx / kx_arr(i,j,k)) * (ez_arr(i+1,j,k) - ez_arr(i,j,k)) -
                    (c.inv_dz / kz_arr(i,j,k)) * (ex_arr(i,j,k+1) - ex_arr(i,j,k)) +
                    phyx(i,j,k) - phyz(i,j,k));
            });
        }
    }

    void step_bz_pml (MultiFab& bz, MultiFab const& ex, MultiFab const& ey,
                      AdiCoeffs const& c, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        MultiFab ky = make_rhs(bz);
        MultiFab kx = make_rhs(bz);
        copy_cc_profile_to_layout(ky, warpx.get_adi_pml_stretch(1), periodicity, 1._rt);
        copy_cc_profile_to_layout(kx, warpx.get_adi_pml_stretch(0), periodicity, 1._rt);
        MultiFab const& psi_hzy = warpx.get_adi_psi_h(AdiPsiH::HZY);
        MultiFab const& psi_hzx = warpx.get_adi_psi_h(AdiPsiH::HZX);
        for (MFIter mfi(bz); mfi.isValid(); ++mfi) {
            auto const bz_arr = bz.array(mfi);
            auto const ex_arr = ex.const_array(mfi);
            auto const ey_arr = ey.const_array(mfi);
            auto const ky_arr = ky.const_array(mfi);
            auto const kx_arr = kx.const_array(mfi);
            auto const phzy = psi_hzy.const_array(mfi);
            auto const phzx = psi_hzx.const_array(mfi);
            Box const& b = mfi.validbox();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bz_arr(i,j,k) += c.dtd2 * (
                    (c.inv_dy / ky_arr(i,j,k)) * (ex_arr(i,j+1,k) - ex_arr(i,j,k)) -
                    (c.inv_dx / kx_arr(i,j,k)) * (ey_arr(i+1,j,k) - ey_arr(i,j,k)) +
                    phzy(i,j,k) - phzx(i,j,k));
            });
        }
    }

    void update_one_psi (
        MultiFab& psi, int deriv_dir, MultiFab const& src,
        IntVect const& off, Real dinv, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        MultiFab acoef = make_rhs(psi);
        MultiFab bcoef = make_rhs(psi);
        copy_cc_profile_to_layout(acoef, warpx.get_adi_pml_a(deriv_dir), periodicity, 0._rt);
        copy_cc_profile_to_layout(bcoef, warpx.get_adi_pml_b(deriv_dir), periodicity, 1._rt);
        for (MFIter mfi(psi); mfi.isValid(); ++mfi) {
            auto const psi_arr = psi.array(mfi);
            auto const a_arr = acoef.const_array(mfi);
            auto const b_arr = bcoef.const_array(mfi);
            auto const src_arr = src.const_array(mfi);
            Box const& bx = mfi.validbox();
            int const oi = off[0];
            int const oj = off[1];
            int const ok = off[2];
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const dnum = src_arr(i+oi, j+oj, k+ok) - src_arr(i,j,k);
                psi_arr(i,j,k) = b_arr(i,j,k) * psi_arr(i,j,k)
                    + a_arr(i,j,k) * dinv * dnum;
            });
        }
        psi.FillBoundary(periodicity);
        psi.setBndry(0._rt);
    }

    void update_psi_e (FieldArray const& Hold, AdiCoeffs const& c,
                       Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EXY), 1, *Hold[2],
                       IntVect(0,-1,0), c.inv_dy, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EXZ), 2, *Hold[1],
                       IntVect(0,0,-1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EYX), 0, *Hold[2],
                       IntVect(-1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EYZ), 2, *Hold[0],
                       IntVect(0,0,-1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EZX), 0, *Hold[1],
                       IntVect(-1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EZY), 1, *Hold[0],
                       IntVect(0,-1,0), c.inv_dy, periodicity);
    }

    void update_psi_h_first (FieldArray const& Enew, MultiFab const& Ex0,
                             MultiFab const& Ey0, MultiFab const& Ez0,
                             AdiCoeffs const& c, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXY), 1, Ez0,
                       IntVect(0,1,0), c.inv_dy, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXZ), 2, *Enew[1],
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYX), 0, *Enew[2],
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYZ), 2, Ex0,
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZX), 0, Ey0,
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZY), 1, *Enew[0],
                       IntVect(0,1,0), c.inv_dy, periodicity);
    }

    void update_psi_h_second (FieldArray const& Enew, MultiFab const& Exh,
                              MultiFab const& Eyh, MultiFab const& Ezh,
                              AdiCoeffs const& c, Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXY), 1, *Enew[2],
                       IntVect(0,1,0), c.inv_dy, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXZ), 2, Eyh,
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYX), 0, Ezh,
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYZ), 2, Exh,
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZX), 0, Eyh,
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZY), 1, *Enew[0],
                       IntVect(0,1,0), c.inv_dy, periodicity);
    }

    void adi_first_half_step (
        FieldArray& Efield,
        FieldArray& Bfield,
        AdiFieldArray& Efield_adi,
        AdiFieldArray& Bfield_adi,
        AdiCoeffs const& c,
        AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity,
        PecConfig const& pec,
        AdiFieldArray const& pec_masks)
    {
        // Implicit E along y,z,x; explicit B at n+1/2.
        MultiFab Ex0 = make_copy(*Efield[0]);
        MultiFab Ey0 = make_copy(*Efield[1]);
        MultiFab Ez0 = make_copy(*Efield[2]);
        FieldArray Hold;
        if (adi_pml_on()) {
            for (int comp = 0; comp < 3; ++comp) {
                Hold[comp] = make_like(*mat.H[comp]);
                MultiFab::Copy(*Hold[comp], *mat.H[comp], 0, 0, 1,
                               mat.H[comp]->nGrowVect());
            }
        }

        copy_fields(Efield_adi[1], Efield, periodicity);
        copy_fields(Bfield_adi[1], mat.H, periodicity);
        MultiFab rhs_ex = build_rhs_ex1(
            *Efield_adi[1][0], *Efield_adi[1][1],
            *Bfield_adi[1][1], *Bfield_adi[1][2], c, mat, periodicity);

        copy_fields(Efield_adi[2], Efield, periodicity);
        copy_fields(Bfield_adi[2], mat.H, periodicity);
        MultiFab rhs_ey = build_rhs_ey1(
            *Efield_adi[2][1], *Efield_adi[2][2],
            *Bfield_adi[2][0], *Bfield_adi[2][2], c, mat, periodicity);

        copy_fields(Efield_adi[0], Efield, periodicity);
        copy_fields(Bfield_adi[0], mat.H, periodicity);
        MultiFab rhs_ez = build_rhs_ez1(
            *Efield_adi[0][2], *Efield_adi[0][0],
            *Bfield_adi[0][0], *Bfield_adi[0][1], c, mat, periodicity);

        solve_implicit_ex1(*Efield_adi[1][0], rhs_ex, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ey1(*Efield_adi[2][1], rhs_ey, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ez1(*Efield_adi[0][2], rhs_ez, c, mat, periodicity, pec, pec_masks);

        copy_field_component(Efield, Efield_adi[1], 0, periodicity);
        copy_field_component(Efield, Efield_adi[2], 1, periodicity);
        copy_field_component(Efield, Efield_adi[0], 2, periodicity);

        fill_boundary_and_sync(Efield, periodicity);
        pin_pec_tangential_e(Efield, pec);

        if (WarpX::use_lumped_inductor == 1) {
            WarpX& warpx = WarpX::GetInstance();
            step_jx(*warpx.get_pointer_current_fp(0, 0),
                    Ex0, *Efield[0], *mat.kappa[0], c);
            step_jy(*warpx.get_pointer_current_fp(0, 1),
                    Ey0, *Efield[1], *mat.kappa[1], c);
            step_jz(*warpx.get_pointer_current_fp(0, 2),
                    Ez0, *Efield[2], *mat.kappa[2], c);
            warpx.FillBoundaryJ(warpx.getngEB());
        }

        if (adi_pml_on()) {
            step_bx_pml(*Bfield[0], *Efield[1], Ez0, c, periodicity);
            step_by_pml(*Bfield[1], *Efield[2], Ex0, c, periodicity);
            step_bz_pml(*Bfield[2], *Efield[0], Ey0, c, periodicity);
            update_psi_e(Hold, c, periodicity);
            update_psi_h_first(Efield, Ex0, Ey0, Ez0, c, periodicity);
        } else {
            step_bx(*Bfield[0], *Efield[1], Ez0, c);
            step_by(*Bfield[1], *Efield[2], Ex0, c);
            step_bz(*Bfield[2], *Efield[0], Ey0, c);
        }

        fill_boundary_and_sync(Bfield, periodicity);
    }

    void adi_second_half_step (
        FieldArray& Efield,
        FieldArray& Bfield,
        AdiFieldArray& Efield_adi,
        AdiFieldArray& Bfield_adi,
        AdiCoeffs const& c,
        AdiMaterialCoeffs const& mat,
        Periodicity const& periodicity,
        PecConfig const& pec,
        AdiFieldArray const& pec_masks)
    {
        // Implicit E along z,x,y; explicit B at n+1.
        MultiFab Exh = make_copy(*Efield[0]);
        MultiFab Eyh = make_copy(*Efield[1]);
        MultiFab Ezh = make_copy(*Efield[2]);
        FieldArray Hold;
        if (adi_pml_on()) {
            for (int comp = 0; comp < 3; ++comp) {
                Hold[comp] = make_like(*mat.H[comp]);
                MultiFab::Copy(*Hold[comp], *mat.H[comp], 0, 0, 1,
                               mat.H[comp]->nGrowVect());
            }
        }

        copy_fields(Efield_adi[2], Efield, periodicity);
        copy_fields(Bfield_adi[2], mat.H, periodicity);
        MultiFab rhs_ex = build_rhs_ex2(
            *Efield_adi[2][0], *Efield_adi[2][2],
            *Bfield_adi[2][1], *Bfield_adi[2][2], c, mat, periodicity);

        copy_fields(Efield_adi[0], Efield, periodicity);
        copy_fields(Bfield_adi[0], mat.H, periodicity);
        MultiFab rhs_ey = build_rhs_ey2(
            *Efield_adi[0][1], *Efield_adi[0][0],
            *Bfield_adi[0][0], *Bfield_adi[0][2], c, mat, periodicity);

        copy_fields(Efield_adi[1], Efield, periodicity);
        copy_fields(Bfield_adi[1], mat.H, periodicity);
        MultiFab rhs_ez = build_rhs_ez2(
            *Efield_adi[1][2], *Efield_adi[1][1],
            *Bfield_adi[1][0], *Bfield_adi[1][1], c, mat, periodicity);

        solve_implicit_ex2(*Efield_adi[2][0], rhs_ex, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ey2(*Efield_adi[0][1], rhs_ey, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ez2(*Efield_adi[1][2], rhs_ez, c, mat, periodicity, pec, pec_masks);

        copy_field_component(Efield, Efield_adi[2], 0, periodicity);
        copy_field_component(Efield, Efield_adi[0], 1, periodicity);
        copy_field_component(Efield, Efield_adi[1], 2, periodicity);

        fill_boundary_and_sync(Efield, periodicity);
        pin_pec_tangential_e(Efield, pec);

        if (WarpX::use_lumped_inductor == 1) {
            WarpX& warpx = WarpX::GetInstance();
            step_jx(*warpx.get_pointer_current_fp(0, 0),
                    Exh, *Efield[0], *mat.kappa[0], c);
            step_jy(*warpx.get_pointer_current_fp(0, 1),
                    Eyh, *Efield[1], *mat.kappa[1], c);
            step_jz(*warpx.get_pointer_current_fp(0, 2),
                    Ezh, *Efield[2], *mat.kappa[2], c);
            warpx.FillBoundaryJ(warpx.getngEB());
        }

        if (adi_pml_on()) {
            step_bx_pml(*Bfield[0], Eyh, *Efield[2], c, periodicity);
            step_by_pml(*Bfield[1], Ezh, *Efield[0], c, periodicity);
            step_bz_pml(*Bfield[2], Exh, *Efield[1], c, periodicity);
            update_psi_e(Hold, c, periodicity);
            update_psi_h_second(Efield, Exh, Eyh, Ezh, c, periodicity);
        } else {
            step_bx(*Bfield[0], Eyh, *Efield[2], c);
            step_by(*Bfield[1], Ezh, *Efield[0], c);
            step_bz(*Bfield[2], Exh, *Efield[1], c);
        }

        fill_boundary_and_sync(Bfield, periodicity);
    }
}

void
FiniteDifferenceSolver::MacroscopicEvolveADI (
    FieldArray& Efield,
    FieldArray& Bfield,
    AdiFieldArray& Efield_adi,
    AdiFieldArray& Bfield_adi,
    AdiFieldArray const& PEC_adi,
    Real const dt,
    Periodicity const& periodicity,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties)
{
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(
        Efield, Bfield, Efield_adi, Bfield_adi, PEC_adi, dt, periodicity,
        macroscopic_properties);
    WARPX_ABORT_WITH_MESSAGE("Macroscopic ADI is implemented only for 3D Cartesian grids.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_fdtd_algo == ElectromagneticSolverAlgo::Yee,
        "Macroscopic ADI currently supports only the Yee solver.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_grid_type == GridType::Staggered,
        "Macroscopic ADI currently supports only staggered Yee fields.");
    PecConfig const pec = get_pec_config(periodicity);
    WarpX::GetInstance().UpdateAdiPmlRecursionCoeffs(dt);

    Real const dx = 1._rt / m_h_stencil_coefs_x[0];
    Real const dy = 1._rt / m_h_stencil_coefs_y[0];
    Real const dz = 1._rt / m_h_stencil_coefs_z[0];

    AdiCoeffs c;
    c.dx = dx;
    c.dy = dy;
    c.dz = dz;
    c.inv_dx = 1._rt / dx;
    c.inv_dy = 1._rt / dy;
    c.inv_dz = 1._rt / dz;
    c.dt = dt;
    c.dtd2 = 0.5_rt * dt;
    c.dtd4 = 0.25_rt * dt;

    pin_pec_tangential_e(Efield, pec);
    apply_pec_mask(Efield);
    fill_boundary_and_sync(Efield, periodicity);
    fill_boundary_and_sync(Bfield, periodicity);

    AdiMaterialCoeffs mat;
    define_material_coeffs(Efield, Bfield, mat);
    update_material_coeffs(mat, Bfield, dt, periodicity, macroscopic_properties);

    WarpX& warpx = WarpX::GetInstance();

    adi_first_half_step(
        Efield, Bfield, Efield_adi, Bfield_adi, c, mat, periodicity, pec, PEC_adi);

    warpx.FillBoundaryE(warpx.getngEB());
    warpx.FillBoundaryB(warpx.getngEB());
    // Soft H/B source once per step, after the first magnetic half-step, at t^{n+1/2}.
    warpx.ApplyExternalFieldExcitationOnGrid(
        ExternalFieldType::BfieldExternal, DtType::FirstHalf, false);

    update_material_coeffs(mat, Bfield, dt, periodicity, macroscopic_properties);
    adi_second_half_step(
        Efield, Bfield, Efield_adi, Bfield_adi, c, mat, periodicity, pec, PEC_adi);

    warpx.FillBoundaryE(warpx.getngEB());
    warpx.FillBoundaryB(warpx.getngEB());
#endif
}
