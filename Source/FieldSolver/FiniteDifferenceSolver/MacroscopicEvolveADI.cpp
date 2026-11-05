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
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

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
        MultiFab const* stretch_e = nullptr,
        MultiFab const* stretch_h = nullptr)
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
            Array4<Real const> se_arr;
            Array4<Real const> sh_arr;
            bool const use_s = stretch_e != nullptr;
            if (use_s) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    stretch_h != nullptr,
                    "ADI PML requires both E-node and H-face stretch.");
                se_arr = stretch_e->const_array(mfi);
                sh_arr = stretch_h->const_array(mfi);
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
                if (use_s) {
                    Real const s_e0 = line_value(se_arr, dir, lo, i, j, k);
                    Real const s_hm1 = line_value(sh_arr, dir, hi - 1, i, j, k);
                    seam_scale = s_e0 * s_hm1;
                }
                Real const alpha = -db_seam * seam_scale * inv_d2;
                Real const beta = -db_seam * seam_scale * inv_d2;

                // Physical coefficients first (needed for Sherman-Morrison corners).
                // Lagged CPML: α,γ use S_E(node) * S_H(face), with S = 1/κ.
                for (int p = 0; p < nsolve; ++p) {
                    int const s = lo + p;
                    Real const db_lo = (p == 0) ? db_seam
                                                : line_value(db_arr, dir, s - 1, i, j, k);
                    Real const db_hi = line_value(db_arr, dir, s, i, j, k);
                    Real s_e = 1._rt;
                    Real s_h_lo = 1._rt;
                    Real s_h_hi = 1._rt;
                    if (use_s) {
                        s_e = line_value(se_arr, dir, s, i, j, k);
                        s_h_lo = (p == 0)
                            ? line_value(sh_arr, dir, hi - 1, i, j, k)
                            : line_value(sh_arr, dir, s - 1, i, j, k);
                        s_h_hi = line_value(sh_arr, dir, s, i, j, k);
                    }
                    Real const al = db_lo * s_e * s_h_lo * inv_d2;
                    Real const ga = db_hi * s_e * s_h_hi * inv_d2;
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
        MultiFab const* stretch_e = nullptr,
        MultiFab const* stretch_h = nullptr)
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
            Array4<Real const> se_arr;
            Array4<Real const> sh_arr;
            bool const use_s = stretch_e != nullptr;
            if (use_s) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    stretch_h != nullptr,
                    "ADI PML requires both E-node and H-face stretch.");
                se_arr = stretch_e->const_array(mfi);
                sh_arr = stretch_h->const_array(mfi);
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
                    Real s_e = 1._rt;
                    Real s_h_lo = 1._rt;
                    Real s_h_hi = 1._rt;
                    if (use_s) {
                        s_e = line_value(se_arr, dir, s, i, j, k);
                        s_h_lo = line_value(sh_arr, dir, s - 1, i, j, k);
                        s_h_hi = line_value(sh_arr, dir, s, i, j, k);
                    }
                    Real const al = db_lo * s_e * s_h_lo * inv_d2;
                    Real const ga = db_hi * s_e * s_h_hi * inv_d2;
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

    // ---------------------------------------------------------------------------------------------
    // Pipelined line solver (warpx.adi_line_solver = pipeline). Each line along `dir` is cut by the
    // block layout into segments, one per box. The Thomas forward sweep runs segment by segment from
    // the low wall up, passing (c', d') of the last row to the next box; the back substitution runs
    // from the high wall down, passing x of the first row back. The arithmetic and its order are those
    // of solve_tridiagonal on the whole line, so the result is bitwise the same as the pencil solve.
    // Lines are split into batches so different boxes along a line work on different batches at once.

    // For every box, the box sharing its low (high) face along dir with the same transverse extent.
    struct LineChain
    {
        BoxArray ba;
        DistributionMapping dm;
        int dir = -1;
        Vector<int> lo_nbr;
        Vector<int> hi_nbr;
    };

    LineChain const& line_chain (MultiFab const& field, int dir)
    {
        static Vector<std::unique_ptr<LineChain>> chains;
        for (auto const& ch : chains) {
            if (ch->dir == dir && ch->ba == field.boxArray() && ch->dm == field.DistributionMap()) {
                return *ch;
            }
        }
        auto ch = std::make_unique<LineChain>();
        ch->ba = field.boxArray();
        ch->dm = field.DistributionMap();
        ch->dir = dir;
        BoxArray const& ba = ch->ba;
        Box const domain = ba.minimalBox();
        int const n = static_cast<int>(ba.size());
        ch->lo_nbr.assign(n, -1);
        ch->hi_nbr.assign(n, -1);
        auto same_transverse = [dir] (Box const& a, Box const& b) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (d == dir) { continue; }
                if (a.smallEnd(d) != b.smallEnd(d) || a.bigEnd(d) != b.bigEnd(d)) { return false; }
            }
            return true;
        };
        for (int b = 0; b < n; ++b) {
            Box const bx = ba[b];
            for (int side = 0; side < 2; ++side) {
                int const plane = (side == 0) ? bx.smallEnd(dir) : bx.bigEnd(dir);
                bool const at_wall = (side == 0) ? (plane == domain.smallEnd(dir))
                                                 : (plane == domain.bigEnd(dir));
                int found = -1;
                int count = 0;
                for (auto const& is : ba.intersections(amrex::makeSlab(bx, dir, plane))) {
                    int const nb = is.first;
                    if (nb == b) { continue; }
                    Box const& other = ba[nb];
                    int const other_plane = (side == 0) ? other.bigEnd(dir) : other.smallEnd(dir);
                    if (other_plane == plane && same_transverse(bx, other)) {
                        found = nb;
                        ++count;
                    }
                }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (at_wall && count == 0) || (!at_wall && count == 1),
                    "warpx.adi_line_solver = pipeline needs a regular grid of boxes (each box face matched by "
                    "exactly one neighbour box); use warpx.adi_line_solver = pencil for this layout.");
                if (side == 0) { ch->lo_nbr[b] = found; } else { ch->hi_nbr[b] = found; }
            }
        }
        chains.push_back(std::move(ch));
        return *chains.back();
    }

    MPI_Comm pipeline_comm ()
    {
        static MPI_Comm comm = MPI_COMM_NULL;
        if (comm == MPI_COMM_NULL) {
            MPI_Comm_dup(ParallelDescriptor::Communicator(), &comm);
            amrex::ExecOnFinalize([] () {
                MPI_Comm c = pipeline_comm();
                MPI_Comm_free(&c);
            });
        }
        return comm;
    }

    void solve_dirichlet_lines_pipelined (
        MultiFab& field, MultiFab const& rhs,
        MultiFab const& Cb, MultiFab const& Db,
        int dir, Real inv_d2,
        MultiFab const* pec_mask,
        MultiFab const* stretch_e = nullptr,
        MultiFab const* stretch_h = nullptr)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field.ixType().nodeCentered(dir),
            "Macroscopic ADI PEC solve expects a nodal line along the implicit direction.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rhs.boxArray() == field.boxArray() &&
            rhs.DistributionMap() == field.DistributionMap(),
            "Pipelined ADI solve expects the right-hand side on the field's layout.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(stretch_e == nullptr || stretch_h != nullptr,
            "ADI PML requires both E-node and H-face stretch.");

        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(dir);
        int const hi = domain.bigEnd(dir);
        int const nsolve = hi - lo - 1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nsolve >= 2,
            "ADI PEC solve needs at least two interior nodes per line.");

        LineChain const& ch = line_chain(field, dir);
        int const nbatch = WarpX::adi_pipeline_batches;
        MPI_Comm const comm = pipeline_comm();
        int const myproc = ParallelDescriptor::MyProc();

        struct Segment
        {
            int gid;
            Box bx;
            int r0;       // first and last row (node index along dir) solved in this box
            int r1;
            int xlo, ylo, zlo, xlen, ylen;
            Long nlines;
            Gpu::DeviceVector<Real> cp, dp;
            Gpu::PinnedVector<Real> fsend, frecv, bsend, brecv;
        };
        // This box owns rows [smallEnd, bigEnd - 1] along dir (the shared top node belongs to the box above),
        // limited to the interior rows lo + 1 .. hi - 1.
        Vector<Segment> segs;
        for (int gid : field.IndexArray()) {
            Segment sg;
            sg.gid = gid;
            sg.bx = field.boxArray()[gid];
            sg.r0 = std::max(sg.bx.smallEnd(dir), lo + 1);
            sg.r1 = std::min(sg.bx.bigEnd(dir) - 1, hi - 1);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(sg.r1 >= sg.r0,
                "Pipelined ADI solve needs every box to own at least one interior row along the solve direction.");
            Box const slab = amrex::makeSlab(sg.bx, dir, sg.bx.smallEnd(dir));
            sg.xlo = slab.smallEnd(0); sg.ylo = slab.smallEnd(1); sg.zlo = slab.smallEnd(2);
            sg.xlen = slab.length(0); sg.ylen = slab.length(1);
            sg.nlines = slab.numPts();
            segs.push_back(std::move(sg));
        }
        // Lower boxes first, so a box whose lower neighbour is on this rank finds its carry already computed.
        std::sort(segs.begin(), segs.end(), [dir] (Segment const& a, Segment const& b) {
            return a.bx.smallEnd(dir) < b.bx.smallEnd(dir);
        });
        Vector<int> seg_of_gid(ch.ba.size(), -1);
        for (int s = 0; s < static_cast<int>(segs.size()); ++s) {
            Segment& sg = segs[s];
            seg_of_gid[sg.gid] = s;
            Long const ncell = sg.nlines * (sg.r1 - sg.r0 + 1);
            sg.cp.resize(ncell);
            sg.dp.resize(ncell);
            if (ch.hi_nbr[sg.gid] >= 0) { sg.fsend.resize(2 * sg.nlines); sg.brecv.resize(sg.nlines); }
            if (ch.lo_nbr[sg.gid] >= 0) { sg.frecv.resize(2 * sg.nlines); sg.bsend.resize(sg.nlines); }
        }
        auto batch_lo = [nbatch] (Long nl, int q) { return (nl * q) / nbatch; };
        auto remote = [&] (int nbr) { return nbr >= 0 && ch.dm[nbr] != myproc; };
        auto tag = [] (int receiver_gid, int phase) { return 2 * receiver_gid + phase; };

        // Post every receive up front; each batch is a separate message, matched in order.
        Vector<MPI_Request> recv_fwd(segs.size() * nbatch, MPI_REQUEST_NULL);
        Vector<MPI_Request> recv_bwd(segs.size() * nbatch, MPI_REQUEST_NULL);
        Vector<MPI_Request> sends;
        for (int s = 0; s < static_cast<int>(segs.size()); ++s) {
            Segment& sg = segs[s];
            for (int q = 0; q < nbatch; ++q) {
                Long const l0 = batch_lo(sg.nlines, q);
                int const cnt = static_cast<int>(batch_lo(sg.nlines, q + 1) - l0);
                if (remote(ch.lo_nbr[sg.gid])) {
                    MPI_Irecv(sg.frecv.data() + 2 * l0, 2 * cnt, ParallelDescriptor::Mpi_typemap<Real>::type(),
                              ch.dm[ch.lo_nbr[sg.gid]], tag(sg.gid, 0), comm, &recv_fwd[s * nbatch + q]);
                }
                if (remote(ch.hi_nbr[sg.gid])) {
                    MPI_Irecv(sg.brecv.data() + l0, cnt, ParallelDescriptor::Mpi_typemap<Real>::type(),
                              ch.dm[ch.hi_nbr[sg.gid]], tag(sg.gid, 1), comm, &recv_bwd[s * nbatch + q]);
                }
            }
        }

        // Forward elimination, batch by batch, lower boxes first.
        for (int q = 0; q < nbatch; ++q) {
            for (int s = 0; s < static_cast<int>(segs.size()); ++s) {
                Segment& sg = segs[s];
                // A box spanning the whole line has nothing to pipeline: solve all its lines in one batch.
                bool const alone = ch.lo_nbr[sg.gid] < 0 && ch.hi_nbr[sg.gid] < 0;
                Long const l0 = alone ? 0 : batch_lo(sg.nlines, q);
                Long const cnt = alone ? (q == 0 ? sg.nlines : 0) : batch_lo(sg.nlines, q + 1) - l0;
                int const lo_nbr = ch.lo_nbr[sg.gid];
                int const hi_nbr = ch.hi_nbr[sg.gid];
                if (remote(lo_nbr)) { MPI_Wait(&recv_fwd[s * nbatch + q], MPI_STATUS_IGNORE); }
                // A lower neighbour on this rank wrote its carry into its own send buffer.
                Real const* carry_in = (lo_nbr < 0) ? nullptr
                    : (remote(lo_nbr) ? sg.frecv.data() : segs[seg_of_gid[lo_nbr]].fsend.data());
                Real* carry_out = (hi_nbr < 0) ? nullptr : sg.fsend.data();

                auto const cb_arr = Cb.const_array(sg.gid);
                auto const db_arr = Db.const_array(sg.gid);
                auto const rhs_arr = rhs.const_array(sg.gid);
                bool const masked = pec_mask != nullptr;
                Array4<Real const> pec_arr;
                if (masked) { pec_arr = pec_mask->const_array(sg.gid); }
                bool const use_s = stretch_e != nullptr;
                Array4<Real const> se_arr;
                Array4<Real const> sh_arr;
                if (use_s) { se_arr = stretch_e->const_array(sg.gid); sh_arr = stretch_h->const_array(sg.gid); }
                Real* cp = sg.cp.data();
                Real* dp = sg.dp.data();
                int const r0 = sg.r0, r1 = sg.r1;
                int const xlo = sg.xlo, ylo = sg.ylo, zlo = sg.zlo, xlen = sg.xlen, ylen = sg.ylen;
                Long const nl = sg.nlines;

                if (cnt > 0) ParallelFor(cnt, [=] AMREX_GPU_DEVICE (Long t) noexcept
                {
                    Long const line = l0 + t;
                    int const i = xlo + static_cast<int>(line % xlen);
                    int const j = ylo + static_cast<int>((line / xlen) % ylen);
                    int const k = zlo + static_cast<int>(line / (static_cast<Long>(xlen) * ylen));
                    Real cprev = 0._rt;
                    Real dprev = 0._rt;
                    if (carry_in != nullptr) {
                        cprev = carry_in[2 * line];
                        dprev = carry_in[2 * line + 1];
                    }
                    for (int s_ = r0; s_ <= r1; ++s_) {
                        int const p = s_ - (lo + 1);
                        Real const db_lo = line_value(db_arr, dir, s_ - 1, i, j, k);
                        Real const db_hi = line_value(db_arr, dir, s_, i, j, k);
                        Real s_e = 1._rt;
                        Real s_h_lo = 1._rt;
                        Real s_h_hi = 1._rt;
                        if (use_s) {
                            s_e = line_value(se_arr, dir, s_, i, j, k);
                            s_h_lo = line_value(sh_arr, dir, s_ - 1, i, j, k);
                            s_h_hi = line_value(sh_arr, dir, s_, i, j, k);
                        }
                        Real const al = db_lo * s_e * s_h_lo * inv_d2;
                        Real const ga = db_hi * s_e * s_h_hi * inv_d2;
                        Real b = 1._rt / line_value(cb_arr, dir, s_, i, j, k) + al + ga;
                        Real a = (p == 0) ? 0._rt : -al;
                        Real c = (p == nsolve - 1) ? 0._rt : -ga;
                        Real r = line_value(rhs_arr, dir, s_, i, j, k);
                        if (masked) {
                            Real const m = line_value(pec_arr, dir, s_, i, j, k);
                            a *= m;
                            b = m * b + (1._rt - m);
                            c *= m;
                            r *= m;
                        }
                        Real cpv;
                        Real dpv;
                        if (p == 0) {
                            Real const denom = b;
                            AMREX_ALWAYS_ASSERT(std::abs(denom) > 0._rt);
                            cpv = c / denom;
                            dpv = r / denom;
                        } else {
                            Real const denom = b - a * cprev;
                            AMREX_ALWAYS_ASSERT(std::abs(denom) > 0._rt);
                            cpv = (p < nsolve - 1) ? c / denom : 0._rt;
                            dpv = (r - a * dprev) / denom;
                        }
                        Long const idx = static_cast<Long>(s_ - r0) * nl + line;
                        cp[idx] = cpv;
                        dp[idx] = dpv;
                        cprev = cpv;
                        dprev = dpv;
                    }
                    if (carry_out != nullptr) {
                        carry_out[2 * line] = cprev;
                        carry_out[2 * line + 1] = dprev;
                    }
                });
                if (remote(hi_nbr)) {
                    Gpu::streamSynchronize();
                    sends.emplace_back();
                    MPI_Isend(sg.fsend.data() + 2 * l0, static_cast<int>(2 * cnt),
                              ParallelDescriptor::Mpi_typemap<Real>::type(),
                              ch.dm[hi_nbr], tag(hi_nbr, 0), comm, &sends.back());
                }
            }
        }

        // Back substitution, batch by batch, upper boxes first.
        for (int q = 0; q < nbatch; ++q) {
            for (int s = static_cast<int>(segs.size()) - 1; s >= 0; --s) {
                Segment& sg = segs[s];
                // A box spanning the whole line has nothing to pipeline: solve all its lines in one batch.
                bool const alone = ch.lo_nbr[sg.gid] < 0 && ch.hi_nbr[sg.gid] < 0;
                Long const l0 = alone ? 0 : batch_lo(sg.nlines, q);
                Long const cnt = alone ? (q == 0 ? sg.nlines : 0) : batch_lo(sg.nlines, q + 1) - l0;
                int const lo_nbr = ch.lo_nbr[sg.gid];
                int const hi_nbr = ch.hi_nbr[sg.gid];
                if (remote(hi_nbr)) { MPI_Wait(&recv_bwd[s * nbatch + q], MPI_STATUS_IGNORE); }
                Real const* x_in = (hi_nbr < 0) ? nullptr
                    : (remote(hi_nbr) ? sg.brecv.data() : segs[seg_of_gid[hi_nbr]].bsend.data());
                Real* x_out = (lo_nbr < 0) ? nullptr : sg.bsend.data();

                auto const field_arr = field.array(sg.gid);
                bool const masked = pec_mask != nullptr;
                Array4<Real const> pec_arr;
                if (masked) { pec_arr = pec_mask->const_array(sg.gid); }
                Real const* cp = sg.cp.data();
                Real const* dp = sg.dp.data();
                int const r0 = sg.r0, r1 = sg.r1;
                int const blo = sg.bx.smallEnd(dir), bhi = sg.bx.bigEnd(dir);
                int const xlo = sg.xlo, ylo = sg.ylo, zlo = sg.zlo, xlen = sg.xlen, ylen = sg.ylen;
                Long const nl = sg.nlines;

                if (cnt > 0) ParallelFor(cnt, [=] AMREX_GPU_DEVICE (Long t) noexcept
                {
                    Long const line = l0 + t;
                    int const i = xlo + static_cast<int>(line % xlen);
                    int const j = ylo + static_cast<int>((line / xlen) % ylen);
                    int const k = zlo + static_cast<int>(line / (static_cast<Long>(xlen) * ylen));
                    Real xnext = 0._rt;
                    if (x_in != nullptr) {
                        // The node shared with the box above: its value was solved there.
                        xnext = x_in[line];
                        Real const m = masked ? line_value(pec_arr, dir, bhi, i, j, k) : 1._rt;
                        set_line_value(field_arr, dir, bhi, i, j, k, masked ? m * xnext : xnext);
                    }
                    for (int s_ = r1; s_ >= r0; --s_) {
                        int const p = s_ - (lo + 1);
                        Long const idx = static_cast<Long>(s_ - r0) * nl + line;
                        Real const x = (p == nsolve - 1) ? dp[idx] : dp[idx] - cp[idx] * xnext;
                        if (masked) {
                            Real const m = line_value(pec_arr, dir, s_, i, j, k);
                            set_line_value(field_arr, dir, s_, i, j, k, m * x);
                        } else {
                            set_line_value(field_arr, dir, s_, i, j, k, x);
                        }
                        xnext = x;
                    }
                    if (x_out != nullptr) { x_out[line] = xnext; }
                    if (blo == lo) { set_line_value(field_arr, dir, lo, i, j, k, 0._rt); }
                    if (bhi == hi) { set_line_value(field_arr, dir, hi, i, j, k, 0._rt); }
                });
                if (remote(lo_nbr)) {
                    Gpu::streamSynchronize();
                    sends.emplace_back();
                    MPI_Isend(sg.bsend.data() + l0, static_cast<int>(cnt),
                              ParallelDescriptor::Mpi_typemap<Real>::type(),
                              ch.dm[lo_nbr], tag(lo_nbr, 1), comm, &sends.back());
                }
            }
        }
        if (!sends.empty()) {
            MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
        }
        // The work and carry buffers are freed on return; the kernels must be done with them.
        Gpu::streamSynchronize();
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

    bool pipeline_on () { return WarpX::adi_use_pipeline; }

    // The field a right-hand side or line solve works on: its pencil copy, or in pipeline mode the block field.
    MultiFab& field_on (AdiFieldArray& pencils, FieldArray const& block, int layout, int comp)
    {
        return pipeline_on() ? *block[comp] : *pencils[layout][comp];
    }

    MultiFab const* pec_mask_on (AdiFieldArray const& pec_masks, int layout, int comp)
    {
        if (!pipeline_on()) { return pec_masks[layout][comp].get(); }
        return WarpX::use_PEC_mask ? WarpX::GetInstance().get_pointer_PEC_fp(0, comp) : nullptr;
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
                                   MultiFab const* stretch_e = nullptr,
                                   MultiFab const* stretch_h = nullptr)
    {
        if (pipeline_on()) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(use_pec_dirichlet_solve(e_comp, solve_dir, pec),
                "Pipelined ADI solve supports only PEC/PML (Dirichlet) lines.");
            solve_dirichlet_lines_pipelined(field, rhs, Cb, Db, solve_dir, inv_d2,
                                            pec_mask, stretch_e, stretch_h);
        } else if (use_pec_dirichlet_solve(e_comp, solve_dir, pec)) {
            solve_dirichlet_nodal_lines(field, rhs, Cb, Db, solve_dir, inv_d2,
                                        pec_mask, stretch_e, stretch_h);
        } else {
            solve_periodic_lines(field, rhs, Cb, Db, solve_dir, inv_d2,
                                 pec_mask, stretch_e, stretch_h);
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

    // PEC termination of in-domain PML: zero B_n on walls normal to the PML
    // direction (matches standalone demo PinPmlNodalBoundaryPlanes on H).
    void pin_pml_normal_b (FieldArray& Bfield, PecConfig const& pec)
    {
        for (int normal = 0; normal < 3; ++normal) {
            if (!pec.normal[normal]) { continue; }

            MultiFab& field = *Bfield[normal];
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
        if (dst.nGrowVect() != IntVect::TheZeroVector()) {
            dst.FillBoundary(periodicity);
        }
    }

    // `src` on the layout of `layout` (0 ghosts and layout's index type when `like` is null, as make_rhs; else
    // like's index type and ghosts, as make_coeff_like). Returns src itself when it already has that layout
    // (pipeline mode), otherwise copies it into `own`.
    MultiFab const& layout_or_copy (
        MultiFab const& layout, MultiFab const* like, MultiFab const& src, MultiFab& own,
        Periodicity const& periodicity)
    {
        BoxArray ba(layout.boxArray());
        IntVect ngrow(0);
        if (like != nullptr) {
            ba.convert(like->ixType());
            ngrow = like->nGrowVect();
        }
        if (src.boxArray() == ba && src.DistributionMap() == layout.DistributionMap() &&
            src.nGrowVect().allGE(ngrow)) {
            return src;
        }
        own.define(ba, layout.DistributionMap(), 1, ngrow, MFInfo().SetArena(The_Async_Arena()));
        copy_coeff_to_layout(own, src, periodicity);
        return own;
    }

    // Material coefficients (Cb, p, kappa, Db) are fixed once dt is fixed, so each one is copied onto a
    // pencil layout once and reused; copying them every half-step was an all-to-all ParallelCopy each time.
    enum class CoeffKind : int { Cb, p, kappa, Db };

    struct CachedCoeff
    {
        CoeffKind kind;
        int comp;
        std::unique_ptr<MultiFab> mf;
    };

    struct PencilCoeffCache
    {
        Real dt = -1._rt;
        bool finalize_registered = false;
        std::vector<CachedCoeff> entries;
    };

    PencilCoeffCache& pencil_coeff_cache ()
    {
        static PencilCoeffCache cache;
        return cache;
    }

    // The coefficient `src` (kind, comp) on the layout of `layout`: with 0 ghosts and layout's index type
    // when `like` is null (as make_rhs), else with like's index type and ghosts (as make_coeff_like).
    MultiFab const& static_coeff_on_layout (
        MultiFab const& layout, MultiFab const* like, CoeffKind kind, int comp,
        MultiFab const& src, Periodicity const& periodicity)
    {
        PencilCoeffCache& cache = pencil_coeff_cache();
        if (!cache.finalize_registered) {
            // Free the cached MultiFabs while AMReX's arenas still exist.
            amrex::ExecOnFinalize([] () { pencil_coeff_cache().entries.clear(); });
            cache.finalize_registered = true;
        }
        BoxArray ba(layout.boxArray());
        IntVect ngrow(0);
        if (like != nullptr) {
            ba.convert(like->ixType());
            ngrow = like->nGrowVect();
        }
        if (src.boxArray() == ba && src.DistributionMap() == layout.DistributionMap() &&
            src.nGrowVect().allGE(ngrow)) {
            return src;  // already on this layout (pipeline mode): no copy, no cache
        }
        for (auto const& e : cache.entries) {
            if (e.kind == kind && e.comp == comp && e.mf->nGrowVect() == ngrow &&
                e.mf->boxArray() == ba && e.mf->DistributionMap() == layout.DistributionMap()) {
                return *e.mf;
            }
        }
        auto mf = std::make_unique<MultiFab>(ba, layout.DistributionMap(), 1, ngrow);
        copy_coeff_to_layout(*mf, src, periodicity);
        cache.entries.push_back(CachedCoeff{kind, comp, std::move(mf)});
        return *cache.entries.back().mf;
    }

    // Fill CFS kappa (0), a (1), b (2), or 1/κ (3) at dst's Yee location.
    void copy_cc_profile_to_layout (
        MultiFab& dst, int dir, int quantity, Periodicity const&)
    {
        FillAdiCfsOnLayout(dst, dir, quantity);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    // Lagged ψ: both implicit and explicit auxiliaries are still at the
    // start of the half-step, so neither is multiplied by b.
    Real pml_elim_psi (
        Real inv_k, Real inv_d, Real q, Real r,
        Real psi_imp_lo, Real psi_imp_hi,
        Real psi_exp_lo, Real psi_exp_hi) noexcept
    {
        return inv_k * inv_d * (
            (r * psi_imp_hi - q * psi_imp_lo)
          - (r * psi_exp_hi - q * psi_exp_lo));
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
            MultiFab const* resistance_mf = WarpX::use_lumped_resistor
                ? macroscopic_properties->m_lumped_resistor_mf[comp].get() : nullptr;

            MultiFab const* capacitance_mf = WarpX::use_lumped_capacitor
                ? macroscopic_properties->m_lumped_capacitor_mf[comp].get() : nullptr;

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
                Array4<Real const> resistance_arr;
                if (resistance_mf != nullptr) {
                    resistance_arr = resistance_mf->const_array(mfi);
                }
                bool const has_resistor = resistance_mf != nullptr;
                Array4<Real const> capacitance_arr;
                if (capacitance_mf != nullptr) {
                    capacitance_arr = capacitance_mf->const_array(mfi);
                }
                bool const has_capacitor = capacitance_mf != nullptr;
                Box const& bx = mfi.tilebox(Cb.ixType().toIntVect());
                auto const& estag = *e_stag[comp];
                Real const scale = kappa_scale[comp];
                bool const has_inductor = inductance_mf != nullptr;
                int const scomp = 0;

                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const sigma_bulk = ablastr::coarsen::sample::Interp(
                        sigma_arr, sigma_stag, estag, macro_cr, i, j, k, scomp);
                    Real const eps_bulk = ablastr::coarsen::sample::Interp(
                        eps_arr, epsilon_stag, estag, macro_cr, i, j, k, scomp);
                    Real const C = has_capacitor ? capacitance_arr(i,j,k) : 0._rt;
                    // J_C = C * edge_length / area * dE/dt adds edge permittivity.
                    Real const eps = eps_bulk + scale * C;
                    Real const R = has_resistor ? resistance_arr(i,j,k) : 0._rt;
                    // Parallel edge conductance: J_R = E * edge_length / (R * area).
                    Real const sigma = sigma_bulk + ((R > 0._rt) ? scale / R : 0._rt);
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

    // Copy components a and b only: each right-hand side reads two E and two H components.
    void copy_fields (FieldArray& dst, FieldArray const& src,
                      Periodicity const& periodicity, int a, int b)
    {
        if (pipeline_on()) { return; }  // the block fields are used in place
        for (int component : {a, b}) {
            dst[component]->ParallelCopy(
                *src[component], 0, 0, 1, IntVect(0),
                dst[component]->nGrowVect(), periodicity);
        }
    }

    void copy_field_component (FieldArray& dst, FieldArray const& src,
                               int component, Periodicity const& periodicity)
    {
        if (pipeline_on()) { return; }  // the block fields are used in place
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
        MultiFab current_field_own;
        MultiFab const& current_field = layout_or_copy(
            rhs, nullptr, current, current_field_own, periodicity);
        MultiFab const& kappa_field = static_coeff_on_layout(
            rhs, nullptr, CoeffKind::kappa, e_comp, kappa, periodicity);

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
        MultiFab const& p_field = static_coeff_on_layout(
            ex, nullptr, CoeffKind::p, 0, *mat.p[0], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ex, nullptr, CoeffKind::Cb, 0, *mat.Cb[0], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ex, mat.Db[2].get(), CoeffKind::Db, 2, *mat.Db[2], periodicity);
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
            // Implicit y: 1/κ_y on Ampere and Faraday. Explicit curls use 1/κ + ψ^n.
            WarpX& warpx = WarpX::GetInstance();
            MultiFab Sy = make_rhs(ex);
            MultiFab kz = make_rhs(ex);
            MultiFab kx_db = make_coeff_like(ex, *mat.Db[2]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ex, nullptr, warpx.get_adi_psi_e(AdiPsiE::EXY), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ex, nullptr, warpx.get_adi_psi_e(AdiPsiE::EXZ), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ex, mat.Db[2].get(), warpx.get_adi_psi_h(AdiPsiH::HZY), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ex, mat.Db[2].get(), warpx.get_adi_psi_h(AdiPsiH::HZX), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sy, 1, 3, periodicity);
            copy_cc_profile_to_layout(kz, 2, 0, periodicity);
            copy_cc_profile_to_layout(kx_db, 0, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sy_arr = Sy.const_array(mfi);
                auto const kz_arr = kz.const_array(mfi);
                auto const kx_arr = kx_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& bx = mfi.validbox();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j-1, k);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sy_arr(i,j,k);
                    Real const curl_h =
                        inv_k * c.inv_dy * (hz_arr(i,j,k) - hz_arr(i,j-1,k)) -
                        (c.inv_dz / kz_arr(i,j,k)) * (hy_arr(i,j,k) - hy_arr(i,j,k-1));
                    Real const ey_lo = ey_arr(i+1,j-1,k) - ey_arr(i,j-1,k);
                    Real const ey_hi = ey_arr(i+1,j,k) - ey_arr(i,j,k);
                    Real const kx_lo = kx_arr(i, j-1, k);
                    Real const kx_hi = kx_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dx * c.inv_dy * ey_lo / kx_lo
                        - inv_k * r * c.inv_dx * c.inv_dy * ey_hi / kx_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dy, q, r,
                                       phimp(i,j-1,k), phimp(i,j,k),
                                       phexp(i,j-1,k), phexp(i,j,k));
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
        MultiFab const& p_field = static_coeff_on_layout(
            ey, nullptr, CoeffKind::p, 1, *mat.p[1], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ey, nullptr, CoeffKind::Cb, 1, *mat.Cb[1], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ey, mat.Db[0].get(), CoeffKind::Db, 0, *mat.Db[0], periodicity);
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
            MultiFab Sz = make_rhs(ey);
            MultiFab kx = make_rhs(ey);
            MultiFab ky_db = make_coeff_like(ey, *mat.Db[0]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ey, nullptr, warpx.get_adi_psi_e(AdiPsiE::EYZ), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ey, nullptr, warpx.get_adi_psi_e(AdiPsiE::EYX), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ey, mat.Db[0].get(), warpx.get_adi_psi_h(AdiPsiH::HXZ), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ey, mat.Db[0].get(), warpx.get_adi_psi_h(AdiPsiH::HXY), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sz, 2, 3, periodicity);
            copy_cc_profile_to_layout(kx, 0, 0, periodicity);
            copy_cc_profile_to_layout(ky_db, 1, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sz_arr = Sz.const_array(mfi);
                auto const kx_arr = kx.const_array(mfi);
                auto const ky_arr = ky_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j, k-1);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sz_arr(i,j,k);
                    Real const curl_h =
                        inv_k * c.inv_dz * (hx_arr(i,j,k) - hx_arr(i,j,k-1)) -
                        (c.inv_dx / kx_arr(i,j,k)) * (hz_arr(i,j,k) - hz_arr(i-1,j,k));
                    Real const ez_lo = ez_arr(i,j+1,k-1) - ez_arr(i,j,k-1);
                    Real const ez_hi = ez_arr(i,j+1,k) - ez_arr(i,j,k);
                    Real const ky_lo = ky_arr(i, j, k-1);
                    Real const ky_hi = ky_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dy * c.inv_dz * ez_lo / ky_lo
                        - inv_k * r * c.inv_dy * c.inv_dz * ez_hi / ky_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dz, q, r,
                                       phimp(i,j,k-1), phimp(i,j,k),
                                       phexp(i,j,k-1), phexp(i,j,k));
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
        MultiFab const& p_field = static_coeff_on_layout(
            ez, nullptr, CoeffKind::p, 2, *mat.p[2], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ez, nullptr, CoeffKind::Cb, 2, *mat.Cb[2], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ez, mat.Db[1].get(), CoeffKind::Db, 1, *mat.Db[1], periodicity);
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
            MultiFab Sx = make_rhs(ez);
            MultiFab ky = make_rhs(ez);
            MultiFab kz_db = make_coeff_like(ez, *mat.Db[1]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ez, nullptr, warpx.get_adi_psi_e(AdiPsiE::EZX), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ez, nullptr, warpx.get_adi_psi_e(AdiPsiE::EZY), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ez, mat.Db[1].get(), warpx.get_adi_psi_h(AdiPsiH::HYX), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ez, mat.Db[1].get(), warpx.get_adi_psi_h(AdiPsiH::HYZ), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sx, 0, 3, periodicity);
            copy_cc_profile_to_layout(ky, 1, 0, periodicity);
            copy_cc_profile_to_layout(kz_db, 2, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sx_arr = Sx.const_array(mfi);
                auto const ky_arr = ky.const_array(mfi);
                auto const kz_arr = kz_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i-1, j, k);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sx_arr(i,j,k);
                    Real const curl_h =
                        inv_k * c.inv_dx * (hy_arr(i,j,k) - hy_arr(i-1,j,k)) -
                        (c.inv_dy / ky_arr(i,j,k)) * (hx_arr(i,j,k) - hx_arr(i,j-1,k));
                    Real const ex_lo = ex_arr(i-1,j,k+1) - ex_arr(i-1,j,k);
                    Real const ex_hi = ex_arr(i,j,k+1) - ex_arr(i,j,k);
                    Real const kz_lo = kz_arr(i-1, j, k);
                    Real const kz_hi = kz_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dz * c.inv_dx * ex_lo / kz_lo
                        - inv_k * r * c.inv_dz * c.inv_dx * ex_hi / kz_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dx, q, r,
                                       phimp(i-1,j,k), phimp(i,j,k),
                                       phexp(i-1,j,k), phexp(i,j,k));
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
        MultiFab const& p_field = static_coeff_on_layout(
            ex, nullptr, CoeffKind::p, 0, *mat.p[0], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ex, nullptr, CoeffKind::Cb, 0, *mat.Cb[0], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ex, mat.Db[1].get(), CoeffKind::Db, 1, *mat.Db[1], periodicity);
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
            MultiFab Sz = make_rhs(ex);
            MultiFab ky = make_rhs(ex);
            MultiFab kx_db = make_coeff_like(ex, *mat.Db[1]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ex, nullptr, warpx.get_adi_psi_e(AdiPsiE::EXY), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ex, nullptr, warpx.get_adi_psi_e(AdiPsiE::EXZ), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ex, mat.Db[1].get(), warpx.get_adi_psi_h(AdiPsiH::HYZ), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ex, mat.Db[1].get(), warpx.get_adi_psi_h(AdiPsiH::HYX), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sz, 2, 3, periodicity);
            copy_cc_profile_to_layout(ky, 1, 0, periodicity);
            copy_cc_profile_to_layout(kx_db, 0, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sz_arr = Sz.const_array(mfi);
                auto const ky_arr = ky.const_array(mfi);
                auto const kx_arr = kx_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j, k-1);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sz_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dy / ky_arr(i,j,k)) * (hz_arr(i,j,k) - hz_arr(i,j-1,k)) -
                        inv_k * c.inv_dz * (hy_arr(i,j,k) - hy_arr(i,j,k-1));
                    Real const ez_lo = ez_arr(i+1,j,k-1) - ez_arr(i,j,k-1);
                    Real const ez_hi = ez_arr(i+1,j,k) - ez_arr(i,j,k);
                    Real const kx_lo = kx_arr(i, j, k-1);
                    Real const kx_hi = kx_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ex_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dx * c.inv_dz * ez_lo / kx_lo
                        - inv_k * r * c.inv_dx * c.inv_dz * ez_hi / kx_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dz, q, r,
                                       phimp(i,j,k-1), phimp(i,j,k),
                                       phexp(i,j,k-1), phexp(i,j,k));
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
        MultiFab const& p_field = static_coeff_on_layout(
            ey, nullptr, CoeffKind::p, 1, *mat.p[1], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ey, nullptr, CoeffKind::Cb, 1, *mat.Cb[1], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ey, mat.Db[2].get(), CoeffKind::Db, 2, *mat.Db[2], periodicity);
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
            MultiFab Sx = make_rhs(ey);
            MultiFab kz = make_rhs(ey);
            MultiFab ky_db = make_coeff_like(ey, *mat.Db[2]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ey, nullptr, warpx.get_adi_psi_e(AdiPsiE::EYZ), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ey, nullptr, warpx.get_adi_psi_e(AdiPsiE::EYX), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ey, mat.Db[2].get(), warpx.get_adi_psi_h(AdiPsiH::HZX), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ey, mat.Db[2].get(), warpx.get_adi_psi_h(AdiPsiH::HZY), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sx, 0, 3, periodicity);
            copy_cc_profile_to_layout(kz, 2, 0, periodicity);
            copy_cc_profile_to_layout(ky_db, 1, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const ex_arr = ex.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hz_arr = hz.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sx_arr = Sx.const_array(mfi);
                auto const kz_arr = kz.const_array(mfi);
                auto const ky_arr = ky_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i-1, j, k);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sx_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dz / kz_arr(i,j,k)) * (hx_arr(i,j,k) - hx_arr(i,j,k-1)) -
                        inv_k * c.inv_dx * (hz_arr(i,j,k) - hz_arr(i-1,j,k));
                    Real const ex_lo = ex_arr(i-1,j+1,k) - ex_arr(i-1,j,k);
                    Real const ex_hi = ex_arr(i,j+1,k) - ex_arr(i,j,k);
                    Real const ky_lo = ky_arr(i-1, j, k);
                    Real const ky_hi = ky_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ey_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dy * c.inv_dx * ex_lo / ky_lo
                        - inv_k * r * c.inv_dy * c.inv_dx * ex_hi / ky_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dx, q, r,
                                       phimp(i-1,j,k), phimp(i,j,k),
                                       phexp(i-1,j,k), phexp(i,j,k));
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
        MultiFab const& p_field = static_coeff_on_layout(
            ez, nullptr, CoeffKind::p, 2, *mat.p[2], periodicity);
        MultiFab const& cb_field = static_coeff_on_layout(
            ez, nullptr, CoeffKind::Cb, 2, *mat.Cb[2], periodicity);
        MultiFab const& db_field = static_coeff_on_layout(
            ez, mat.Db[0].get(), CoeffKind::Db, 0, *mat.Db[0], periodicity);
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
            MultiFab Sy = make_rhs(ez);
            MultiFab kx = make_rhs(ez);
            MultiFab kz_db = make_coeff_like(ez, *mat.Db[0]);
            MultiFab psi_e0_own;
            MultiFab const& psi_e0 = layout_or_copy(
                ez, nullptr, warpx.get_adi_psi_e(AdiPsiE::EZX), psi_e0_own, periodicity);
            MultiFab psi_e1_own;
            MultiFab const& psi_e1 = layout_or_copy(
                ez, nullptr, warpx.get_adi_psi_e(AdiPsiE::EZY), psi_e1_own, periodicity);
            MultiFab psi_himp_own;
            MultiFab const& psi_himp = layout_or_copy(
                ez, mat.Db[0].get(), warpx.get_adi_psi_h(AdiPsiH::HXY), psi_himp_own, periodicity);
            MultiFab psi_hexp_own;
            MultiFab const& psi_hexp = layout_or_copy(
                ez, mat.Db[0].get(), warpx.get_adi_psi_h(AdiPsiH::HXZ), psi_hexp_own, periodicity);
            copy_cc_profile_to_layout(Sy, 1, 3, periodicity);
            copy_cc_profile_to_layout(kx, 0, 0, periodicity);
            copy_cc_profile_to_layout(kz_db, 2, 0, periodicity);
            for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const rhs_arr = rhs.array(mfi);
                auto const ez_arr = ez.const_array(mfi);
                auto const ey_arr = ey.const_array(mfi);
                auto const hx_arr = hx.const_array(mfi);
                auto const hy_arr = hy.const_array(mfi);
                auto const p_arr = p_field.const_array(mfi);
                auto const db_arr = db_field.const_array(mfi);
                auto const Sy_arr = Sy.const_array(mfi);
                auto const kx_arr = kx.const_array(mfi);
                auto const kz_arr = kz_db.const_array(mfi);
                auto const pe0 = psi_e0.const_array(mfi);
                auto const pe1 = psi_e1.const_array(mfi);
                auto const phimp = psi_himp.const_array(mfi);
                auto const phexp = psi_hexp.const_array(mfi);
                Box const& b = mfi.validbox();
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const q = db_arr(i, j-1, k);
                    Real const r = db_arr(i, j, k);
                    Real const inv_k = Sy_arr(i,j,k);
                    Real const curl_h =
                        (c.inv_dx / kx_arr(i,j,k)) * (hy_arr(i,j,k) - hy_arr(i-1,j,k)) -
                        inv_k * c.inv_dy * (hx_arr(i,j,k) - hx_arr(i,j-1,k));
                    Real const ey_lo = ey_arr(i,j-1,k+1) - ey_arr(i,j-1,k);
                    Real const ey_hi = ey_arr(i,j,k+1) - ey_arr(i,j,k);
                    Real const kz_lo = kz_arr(i, j-1, k);
                    Real const kz_hi = kz_arr(i, j, k);
                    rhs_arr(i,j,k) = p_arr(i,j,k) * ez_arr(i,j,k) + curl_h
                        + inv_k * q * c.inv_dz * c.inv_dy * ey_lo / kz_lo
                        - inv_k * r * c.inv_dz * c.inv_dy * ey_hi / kz_hi
                        + pe0(i,j,k) - pe1(i,j,k)
                        + pml_elim_psi(inv_k, c.inv_dy, q, r,
                                       phimp(i,j-1,k), phimp(i,j,k),
                                       phexp(i,j-1,k), phexp(i,j,k));
                });
            }
        }
        add_lumped_inductor_current_to_rhs(
            rhs, *mat.kappa[2], 2, periodicity);
        add_soft_e_source_to_rhs(rhs, cb_field, 2, 2);
        return rhs;
    }

    // Nonlinear centered Josephson solve on an existing ADI pencil layout.
    // F(E) = A E + J(E) - rhs, with phi(E) = phi_old + dt*eta/4*(E_old+E).
    // Each iteration solves (A + J_J(E)) dE = -F(E) and sets E := E + dE.
    // J_J changes only the diagonal. Convergence compares ||dE||_inf with
    // newton_atol + newton_rtol * max(||E_old||, ||E||).
    void solve_josephson_component (
        MultiFab& field, MultiFab const& rhs, MultiFab const& Cb, MultiFab const& Db,
        int e_comp, int solve_dir, Real inv_d2, PecConfig const& pec,
        MultiFab const* pec_mask, MultiFab const* stretch_e, MultiFab const* stretch_h,
        Real dt, Periodicity const& periodicity)
    {
        auto linear_solve = [&](MultiFab& out, MultiFab const& source, MultiFab const& cb) {
            solve_implicit_component(out, source, cb, Db, e_comp, solve_dir,
                                     inv_d2, pec, pec_mask, stretch_e, stretch_h);
        };
        if (!WarpX::use_josephson_junction) {
            linear_solve(field, rhs, Cb);
            return;
        }
        auto& warpx = WarpX::GetInstance();
        auto& macro = warpx.GetMacroscopicProperties();
        MultiFab old_e = make_rhs(field);
        MultiFab old_phi = make_rhs(field);
        MultiFab jc = make_rhs(field);
        MultiFab source = make_rhs(field);
        MultiFab cb_newton = make_rhs(field);
        MultiFab updated = make_rhs(field);
        MultiFab delta = make_rhs(field);
        MultiFab::Copy(old_e, field, 0, 0, 1, 0);
        copy_coeff_to_layout(old_phi, *macro.m_jj_phi[e_comp], periodicity);
        copy_coeff_to_layout(jc, *macro.m_jj_Ic[e_comp], periodicity);
        auto const dx = warpx.Geom(0).CellSizeArray();
        Real const length = dx[e_comp];
        Real const area = dx[(e_comp+1)%3] * dx[(e_comp+2)%3];
        Real const phase_factor = dt * 0.5_rt * PhysConst::q_e / PhysConst::hbar * length;
        jc.mult(1._rt / area, 0, 1, 0);
        if (jc.norm0(0) == 0._rt) {
            linear_solve(field, rhs, Cb);
            return;
        }

        // (A + J_J) E_new = rhs - J(E) + J_J E, so E_new = E + dE with
        // (A + J_J) dE = -F(E). The line solver returns E_new.
        auto assemble = [&](MultiFab const& iterate) {
            for (MFIter mfi(source); mfi.isValid(); ++mfi) {
                auto const r = rhs.const_array(mfi);
                auto const e = iterate.const_array(mfi);
                auto const e0 = old_e.const_array(mfi);
                auto const p0 = old_phi.const_array(mfi);
                auto const j = jc.const_array(mfi);
                auto const cb = Cb.const_array(mfi);
                auto const out = source.array(mfi);
                auto const cn = cb_newton.array(mfi);
                Array4<Real const> mask;
                if (pec_mask) { mask = pec_mask->const_array(mfi); }
                bool const masked = pec_mask != nullptr;
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int jj, int k) noexcept {
                    Real const critical = (masked && mask(i,jj,k) == 0._rt) ? 0._rt : j(i,jj,k);
                    Real const phase = p0(i,jj,k) + phase_factor * (e0(i,jj,k) + e(i,jj,k));
                    Real const current =
                        0.5_rt * critical * (std::sin(p0(i,jj,k)) + std::sin(phase));
                    Real const derivative = 0.5_rt * critical * phase_factor * std::cos(phase);
                    Real const diagonal = 1._rt / cb(i,jj,k) + derivative;
                    // Preserve positive definiteness for the unpivoted line solver.
                    // This is a conservative robustness condition, not a CFL proof.
                    cn(i,jj,k) = diagonal;
                    out(i,jj,k) = r(i,jj,k) - current + derivative * e(i,jj,k);
                });
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!source.contains_nan() && !source.contains_inf() &&
                !cb_newton.contains_nan() && !cb_newton.contains_inf() && cb_newton.min(0) > 0._rt,
                "Josephson Newton diagonal is nonpositive or nonfinite; reduce warpx.const_dt.");
            cb_newton.invert(1._rt, 0, 1, 0);
        };
        bool converged = false;
        for (int iteration = 0; iteration < macro.m_jj_max_iterations; ++iteration) {
            assemble(field);
            linear_solve(updated, source, cb_newton);
            MultiFab::LinComb(delta, 1._rt, updated, 0, -1._rt, field, 0, 0, 1, 0);
            MultiFab::Add(field, delta, 0, 0, 1, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!delta.contains_nan() && !delta.contains_inf(),
                "Nonfinite Josephson Newton step; reduce warpx.const_dt.");
            Real const step = delta.norm0(0);
            Real const tolerance = macro.m_jj_atol + macro.m_jj_rtol *
                std::max(old_e.norm0(0), field.norm0(0));
            if (step <= tolerance) {
                converged = true;
                break;
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(converged,
            "Josephson Newton failed to converge; reduce warpx.const_dt or increase josephson.max_iterations.");

        // Commit phase once, from the saved half-step state. Do not deposit Js
        // into persistent J: the centered supercurrent was included in the solve.
        for (MFIter mfi(old_phi); mfi.isValid(); ++mfi) {
            auto const p = old_phi.array(mfi);
            auto const e = field.const_array(mfi);
            auto const e0 = old_e.const_array(mfi);
            auto const j = jc.const_array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int jj, int k) noexcept {
                if (j(i,jj,k) != 0._rt) {
                    p(i,jj,k) += phase_factor * (e0(i,jj,k) + e(i,jj,k));
                }
            });
        }
        macro.m_jj_phi[e_comp]->ParallelCopy(old_phi, 0, 0, 1, IntVect(0), IntVect(0), periodicity);
        macro.m_jj_phi[e_comp]->FillBoundaryAndSync(periodicity);
    }

    void solve_implicit_ex1 (MultiFab& ex, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ex, nullptr, CoeffKind::Cb, 0, *mat.Cb[0], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ex, mat.Db[2].get(), CoeffKind::Db, 2, *mat.Db[2], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 1, 0);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ex);
            stretch_h_mf = make_coeff_like(ex, *mat.Db[2]);
            copy_cc_profile_to_layout(stretch_e_mf, 1, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 1, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ex, rhs, Cb, Db,
                                 0, 1, c.inv_dy * c.inv_dy, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
    }

    void solve_implicit_ey1 (MultiFab& ey, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ey, nullptr, CoeffKind::Cb, 1, *mat.Cb[1], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ey, mat.Db[0].get(), CoeffKind::Db, 0, *mat.Db[0], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 2, 1);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ey);
            stretch_h_mf = make_coeff_like(ey, *mat.Db[0]);
            copy_cc_profile_to_layout(stretch_e_mf, 2, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 2, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ey, rhs, Cb, Db,
                                 1, 2, c.inv_dz * c.inv_dz, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
    }

    void solve_implicit_ez1 (MultiFab& ez, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ez, nullptr, CoeffKind::Cb, 2, *mat.Cb[2], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ez, mat.Db[1].get(), CoeffKind::Db, 1, *mat.Db[1], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 0, 2);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ez);
            stretch_h_mf = make_coeff_like(ez, *mat.Db[1]);
            copy_cc_profile_to_layout(stretch_e_mf, 0, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 0, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ez, rhs, Cb, Db,
                                 2, 0, c.inv_dx * c.inv_dx, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
    }

    void solve_implicit_ex2 (MultiFab& ex, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ex, nullptr, CoeffKind::Cb, 0, *mat.Cb[0], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ex, mat.Db[1].get(), CoeffKind::Db, 1, *mat.Db[1], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 2, 0);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ex);
            stretch_h_mf = make_coeff_like(ex, *mat.Db[1]);
            copy_cc_profile_to_layout(stretch_e_mf, 2, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 2, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ex, rhs, Cb, Db,
                                 0, 2, c.inv_dz * c.inv_dz, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
    }

    void solve_implicit_ey2 (MultiFab& ey, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ey, nullptr, CoeffKind::Cb, 1, *mat.Cb[1], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ey, mat.Db[2].get(), CoeffKind::Db, 2, *mat.Db[2], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 0, 1);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ey);
            stretch_h_mf = make_coeff_like(ey, *mat.Db[2]);
            copy_cc_profile_to_layout(stretch_e_mf, 0, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 0, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ey, rhs, Cb, Db,
                                 1, 0, c.inv_dx * c.inv_dx, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
    }

    void solve_implicit_ez2 (MultiFab& ez, MultiFab const& rhs,
                             AdiCoeffs const& c, AdiMaterialCoeffs const& mat,
                             Periodicity const& periodicity, PecConfig const& pec,
                             AdiFieldArray const& pec_masks)
    {
        MultiFab const& Cb = static_coeff_on_layout(
            ez, nullptr, CoeffKind::Cb, 2, *mat.Cb[2], periodicity);
        MultiFab const& Db = static_coeff_on_layout(
            ez, mat.Db[0].get(), CoeffKind::Db, 0, *mat.Db[0], periodicity);
        MultiFab const* pec_mask = pec_mask_on(pec_masks, 1, 2);
        MultiFab stretch_e_mf, stretch_h_mf;
        MultiFab const* stretch_e = nullptr;
        MultiFab const* stretch_h = nullptr;
        if (adi_pml_on()) {
            stretch_e_mf = make_rhs(ez);
            stretch_h_mf = make_coeff_like(ez, *mat.Db[0]);
            copy_cc_profile_to_layout(stretch_e_mf, 1, 3, periodicity);
            copy_cc_profile_to_layout(stretch_h_mf, 1, 3, periodicity);
            stretch_e = &stretch_e_mf;
            stretch_h = &stretch_h_mf;
        }
        solve_josephson_component(ez, rhs, Cb, Db,
                                 2, 1, c.inv_dy * c.inv_dy, pec, pec_mask,
                                 stretch_e, stretch_h, c.dt, periodicity);
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
        copy_cc_profile_to_layout(kz, 2, 0, periodicity);
        copy_cc_profile_to_layout(ky, 1, 0, periodicity);
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
        copy_cc_profile_to_layout(kx, 0, 0, periodicity);
        copy_cc_profile_to_layout(kz, 2, 0, periodicity);
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
        copy_cc_profile_to_layout(ky, 1, 0, periodicity);
        copy_cc_profile_to_layout(kx, 0, 0, periodicity);
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
        copy_cc_profile_to_layout(acoef, deriv_dir, 1, periodicity);
        copy_cc_profile_to_layout(bcoef, deriv_dir, 2, periodicity);
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
        psi.setBndry(0._rt);
        psi.FillBoundary(periodicity);
    }

    // Electric auxiliaries use the same Yee difference as Ampere's curl:
    // dnum = H(i,j,k) - H(i-off), implemented as update_one_psi(..., -off, -inv).
    // Gedney ADI-CPML updates every electric auxiliary from the completed H
    // state at the end of the half-step.  The explicit/implicit split belongs
    // to the field update, not to the subsequent convolution recursion.
    void update_psi_e (FieldArray const& Hnew, AdiCoeffs const& c,
                       Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EXY), 1, *Hnew[2],
                       IntVect(0,-1,0), -c.inv_dy, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EXZ), 2, *Hnew[1],
                       IntVect(0,0,-1), -c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EYX), 0, *Hnew[2],
                       IntVect(-1,0,0), -c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EYZ), 2, *Hnew[0],
                       IntVect(0,0,-1), -c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EZX), 0, *Hnew[1],
                       IntVect(-1,0,0), -c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_e(AdiPsiE::EZY), 1, *Hnew[0],
                       IntVect(0,-1,0), -c.inv_dy, periodicity);
    }

    // Likewise, all magnetic auxiliaries use the completed E state at the end
    // of the half-step.  Field equations consume the old auxiliaries first.
    void update_psi_h (FieldArray const& Enew, AdiCoeffs const& c,
                       Periodicity const& periodicity)
    {
        WarpX& warpx = WarpX::GetInstance();
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXY), 1, *Enew[2],
                       IntVect(0,1,0), c.inv_dy, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HXZ), 2, *Enew[1],
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYX), 0, *Enew[2],
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HYZ), 2, *Enew[0],
                       IntVect(0,0,1), c.inv_dz, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZX), 0, *Enew[1],
                       IntVect(1,0,0), c.inv_dx, periodicity);
        update_one_psi(warpx.get_adi_psi_h(AdiPsiH::HZY), 1, *Enew[0],
                       IntVect(0,1,0), c.inv_dy, periodicity);
    }

    void fill_h_from_b (FieldArray& H, FieldArray const& B,
                        AdiMaterialCoeffs const& mat, AdiCoeffs const& c,
                        Periodicity const& periodicity)
    {
        for (int comp = 0; comp < 3; ++comp) {
            MultiFab& h = *H[comp];
            MultiFab const& b = *B[comp];
            MultiFab const& db = *mat.Db[comp];
            for (MFIter mfi(h); mfi.isValid(); ++mfi) {
                auto const h_arr = h.array(mfi);
                auto const b_arr = b.const_array(mfi);
                auto const db_arr = db.const_array(mfi);
                Box const& bx = mfi.tilebox();
                Real const dtd2 = c.dtd2;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    h_arr(i,j,k) = b_arr(i,j,k) * db_arr(i,j,k) / dtd2;
                });
            }
        }
        fill_boundary_and_sync(H, periodicity);
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

        copy_fields(Efield_adi[1], Efield, periodicity, 0, 1);
        copy_fields(Bfield_adi[1], mat.H, periodicity, 1, 2);
        MultiFab rhs_ex = build_rhs_ex1(
            field_on(Efield_adi, Efield, 1, 0), field_on(Efield_adi, Efield, 1, 1),
            field_on(Bfield_adi, mat.H, 1, 1), field_on(Bfield_adi, mat.H, 1, 2), c, mat, periodicity);

        copy_fields(Efield_adi[2], Efield, periodicity, 1, 2);
        copy_fields(Bfield_adi[2], mat.H, periodicity, 0, 2);
        MultiFab rhs_ey = build_rhs_ey1(
            field_on(Efield_adi, Efield, 2, 1), field_on(Efield_adi, Efield, 2, 2),
            field_on(Bfield_adi, mat.H, 2, 0), field_on(Bfield_adi, mat.H, 2, 2), c, mat, periodicity);

        copy_fields(Efield_adi[0], Efield, periodicity, 2, 0);
        copy_fields(Bfield_adi[0], mat.H, periodicity, 0, 1);
        MultiFab rhs_ez = build_rhs_ez1(
            field_on(Efield_adi, Efield, 0, 2), field_on(Efield_adi, Efield, 0, 0),
            field_on(Bfield_adi, mat.H, 0, 0), field_on(Bfield_adi, mat.H, 0, 1), c, mat, periodicity);

        solve_implicit_ex1(field_on(Efield_adi, Efield, 1, 0), rhs_ex, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ey1(field_on(Efield_adi, Efield, 2, 1), rhs_ey, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ez1(field_on(Efield_adi, Efield, 0, 2), rhs_ez, c, mat, periodicity, pec, pec_masks);

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
            // Particle-free runs can allocate fewer current than E/B guards.
            warpx.FillBoundaryJ(warpx.get_pointer_current_fp(0, 0)->nGrowVect());
        }

        if (adi_pml_on()) {
            // Field updates use ψ from the start of the half-step.  After all
            // fields reach n+1/2, advance every auxiliary from that new state.
            step_bx_pml(*Bfield[0], *Efield[1], Ez0, c, periodicity);
            step_by_pml(*Bfield[1], *Efield[2], Ex0, c, periodicity);
            step_bz_pml(*Bfield[2], *Efield[0], Ey0, c, periodicity);
            fill_h_from_b(Hold, Bfield, mat, c, periodicity);
            update_psi_e(Hold, c, periodicity);
            update_psi_h(Efield, c, periodicity);
        } else {
            step_bx(*Bfield[0], *Efield[1], Ez0, c);
            step_by(*Bfield[1], *Efield[2], Ex0, c);
            step_bz(*Bfield[2], *Efield[0], Ey0, c);
        }

        fill_boundary_and_sync(Bfield, periodicity);
        pin_pml_normal_b(Bfield, pec);
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

        copy_fields(Efield_adi[2], Efield, periodicity, 0, 2);
        copy_fields(Bfield_adi[2], mat.H, periodicity, 1, 2);
        MultiFab rhs_ex = build_rhs_ex2(
            field_on(Efield_adi, Efield, 2, 0), field_on(Efield_adi, Efield, 2, 2),
            field_on(Bfield_adi, mat.H, 2, 1), field_on(Bfield_adi, mat.H, 2, 2), c, mat, periodicity);

        copy_fields(Efield_adi[0], Efield, periodicity, 1, 0);
        copy_fields(Bfield_adi[0], mat.H, periodicity, 0, 2);
        MultiFab rhs_ey = build_rhs_ey2(
            field_on(Efield_adi, Efield, 0, 1), field_on(Efield_adi, Efield, 0, 0),
            field_on(Bfield_adi, mat.H, 0, 0), field_on(Bfield_adi, mat.H, 0, 2), c, mat, periodicity);

        copy_fields(Efield_adi[1], Efield, periodicity, 2, 1);
        copy_fields(Bfield_adi[1], mat.H, periodicity, 0, 1);
        MultiFab rhs_ez = build_rhs_ez2(
            field_on(Efield_adi, Efield, 1, 2), field_on(Efield_adi, Efield, 1, 1),
            field_on(Bfield_adi, mat.H, 1, 0), field_on(Bfield_adi, mat.H, 1, 1), c, mat, periodicity);

        solve_implicit_ex2(field_on(Efield_adi, Efield, 2, 0), rhs_ex, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ey2(field_on(Efield_adi, Efield, 0, 1), rhs_ey, c, mat, periodicity, pec, pec_masks);
        solve_implicit_ez2(field_on(Efield_adi, Efield, 1, 2), rhs_ez, c, mat, periodicity, pec, pec_masks);

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
            // Particle-free runs can allocate fewer current than E/B guards.
            warpx.FillBoundaryJ(warpx.get_pointer_current_fp(0, 0)->nGrowVect());
        }

        if (adi_pml_on()) {
            // Repeat the Gedney ordering at n+1: fields first, then all ψ.
            step_bx_pml(*Bfield[0], Eyh, *Efield[2], c, periodicity);
            step_by_pml(*Bfield[1], Ezh, *Efield[0], c, periodicity);
            step_bz_pml(*Bfield[2], Exh, *Efield[1], c, periodicity);
            fill_h_from_b(Hold, Bfield, mat, c, periodicity);
            update_psi_e(Hold, c, periodicity);
            update_psi_h(Efield, c, periodicity);
        } else {
            step_bx(*Bfield[0], Eyh, *Efield[2], c);
            step_by(*Bfield[1], Ezh, *Efield[0], c);
            step_bz(*Bfield[2], Exh, *Efield[1], c);
        }

        fill_boundary_and_sync(Bfield, periodicity);
        pin_pml_normal_b(Bfield, pec);
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

    // The cached pencil coefficients depend on dt; drop them if dt changes.
    if (dt != pencil_coeff_cache().dt) {
        pencil_coeff_cache().entries.clear();
        pencil_coeff_cache().dt = dt;
    }

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
