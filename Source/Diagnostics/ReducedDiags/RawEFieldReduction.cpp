/* Copyright 2021 Revathi Jambunathan, Saurabh Sawant, Andrew Nonaka
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "RawEFieldReduction.H"

#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXUtil.H"

#include <AMReX_Algorithm.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <limits>
#include <ostream>

#include <regex>

namespace
{
    // The smallest sub-box of `box` that holds every point where `parser` is nonzero, or an
    // empty box. Points are placed exactly as in ComputeRawEFieldReduction, so the two agree on
    // which points are nonzero.
    amrex::Box NonzeroBounds (
        amrex::Box const& box, amrex::IntVect const& nodalType,
        amrex::ParserExecutor<3> const& parser,
        amrex::RealBox const& real_box,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
    {
        using namespace amrex::literals;
        int const imax = std::numeric_limits<int>::max();
        int const imin = std::numeric_limits<int>::lowest();

        amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMin, amrex::ReduceOpMin,
                         amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax> rop;
        amrex::ReduceData<int, int, int, int, int, int> rdata(rop);
        using RT = typename decltype(rdata)::Type;
        rop.eval(box, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
        {
            amrex::Real fac_x = (1._rt - nodalType[0]) * dx[0] * 0.5_rt;
            amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==1)
            amrex::Real y = 0._rt;
            amrex::Real z = 0._rt;
#elif (AMREX_SPACEDIM==2)
            amrex::Real y = 0._rt;
            amrex::Real fac_z = (1._rt - nodalType[1]) * dx[1] * 0.5_rt;
            amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
            amrex::Real fac_y = (1._rt - nodalType[1]) * dx[1] * 0.5_rt;
            amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
            amrex::Real fac_z = (1._rt - nodalType[2]) * dx[2] * 0.5_rt;
            amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
            if (parser(x,y,z) != 0._rt) {
                return {i, j, k, i, j, k};
            }
            return {imax, imax, imax, imin, imin, imin};
        });
        auto const r = rdata.value(rop);
        if (amrex::get<0>(r) > amrex::get<3>(r)) {
            return amrex::Box();
        }
        amrex::IntVect const lo(AMREX_D_DECL(amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r)));
        amrex::IntVect const hi(AMREX_D_DECL(amrex::get<3>(r), amrex::get<4>(r), amrex::get<5>(r)));
        return amrex::Box(lo, hi, amrex::IndexType(nodalType));
    }
}

//constructor
RawEFieldReduction::RawEFieldReduction (std::string rd_name)
: ReducedDiags{rd_name}
{
    using namespace amrex::literals;

    // RZ coordinate is not working
#if (defined WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "RawEFieldReduction reduced diagnostics does not work for RZ coordinate.");
#endif

    // read number of levels
    int nLevel = 0;
    amrex::ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nLevel == 0,
        "RawEFieldReduction reduced diagnostics does not work with mesh refinement.");

    constexpr int noutputs = 3; // Three outputs in the Raw Efield reduction diagnostic (Ex, Ey, Ez)
    // resize data array
    m_data.resize(noutputs, 0.0_rt);

    amrex::ParmParse pp_rd_name(rd_name);

    // read reduced function with parser
    std::string parser_string = "";
    utils::parser::Store_parserString(pp_rd_name,"reduced_function(x,y,z)", parser_string);
    m_parser = std::make_unique<amrex::Parser>(utils::parser::makeParser(parser_string,{"x","y","z"}));

    // Replace all newlines and possible following whitespaces with a single whitespace. This
    // should avoid weird formatting when the string is written in the header of the output file.
    parser_string = std::regex_replace(parser_string, std::regex("\n\\s*"), " ");

    // read reduction type
    std::string reduction_type_string;
    pp_rd_name.get("reduction_type", reduction_type_string);
    m_reduction_type = GetAlgorithmInteger (pp_rd_name, "reduction_type");
    if (m_reduction_type == ReductionType::Sum) {
        m_integral_type = GetAlgorithmInteger (pp_rd_name, "integration_type");
    }

    if (m_integral_type == IntegrationType::Surface) {
        std::string surface_normal_string;
        pp_rd_name.get("surface_normal", surface_normal_string);
        if (surface_normal_string == "x" || surface_normal_string == "X") {
            m_surface_normal[0] = 1;
        }
#if (AMREX_SPACEDIM==1)
        else if (surface_normal_string == "y" || surface_normal_string == "Y") {
            amrex::Abort("In 1-D, we compute over an X line. So the only direction for the surface integral is X.");
        }
        else if (surface_normal_string == "z" || surface_normal_string == "Z") {
            amrex::Abort("In 1-D, we compute over an X line. So the only direction for the surface integral is X.");
        }
#elif (AMREX_SPACEDIM==2)
        else if (surface_normal_string == "y" || surface_normal_string == "Y") {
            amrex::Abort("In 2-D, we compute over an X-Z plane. So the plane of interest for the surface integral is Z.");
        }
        else if (surface_normal_string == "z" || surface_normal_string == "Z") {
            m_surface_normal[1] = 1;
        }
#else
        else if (surface_normal_string == "y" || surface_normal_string == "Y") {
            m_surface_normal[1] = 1;
        }
        else if (surface_normal_string == "z" || surface_normal_string == "Z") {
            m_surface_normal[2] = 1;
        }
#endif
        pp_rd_name.queryarr("scaling_factor", m_scaling_factor, 0, AMREX_SPACEDIM);
        AMREX_ASSERT(m_scaling_factor.size() == AMREX_SPACEDIM);
    }

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if ( m_IsNotRestart )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]" + reduction_type_string + " of " + parser_string + " (SI units)";

            ofs << std::endl;
            // close file
            ofs.close();
        }
    }
}

void RawEFieldReduction::UpdateNonzeroBounds ()
{
    auto & warpx = WarpX::GetInstance();
    const int lev = 0;
    const amrex::MultiFab &Ex = warpx.getEfield(lev,0);
    const amrex::MultiFab &Ey = warpx.getEfield(lev,1);
    const amrex::MultiFab &Ez = warpx.getEfield(lev,2);
    if (!m_nonzero_bounds.empty() &&
        m_bounds_ba == Ex.boxArray() && m_bounds_dm == Ex.DistributionMap()) {
        return;
    }
    m_bounds_ba = Ex.boxArray();
    m_bounds_dm = Ex.DistributionMap();
    m_nonzero_bounds.assign(Ex.local_size(), std::array<amrex::Box,3>{});

    auto reduction_function_parser = m_parser->compile<m_nvars>();
    amrex::Geometry const & geom = warpx.Geom(lev);
    const amrex::RealBox& real_box = geom.ProbDomain();
    const auto dx = geom.CellSizeArray();
    const amrex::IntVect nodal[3] = {
        Ex.ixType().toIntVect(), Ey.ixType().toIntVect(), Ez.ixType().toIntVect()};

    for ( amrex::MFIter mfi(Ex, false); mfi.isValid(); ++mfi)
    {
        auto& bounds = m_nonzero_bounds[mfi.LocalIndex()];
        for (int comp = 0; comp < 3; ++comp) {
            bounds[comp] = NonzeroBounds(mfi.tilebox(nodal[comp]), nodal[comp],
                                         reduction_function_parser, real_box, dx);
        }
    }
}

// function that does an arbitrary reduction of the electromagnetic fields
void RawEFieldReduction::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    if (m_reduction_type == ReductionType::Maximum)
    {
        ComputeRawEFieldReduction<amrex::ReduceOpMax>();
    }
    else if (m_reduction_type == ReductionType::Minimum)
    {
        ComputeRawEFieldReduction<amrex::ReduceOpMin>();
    }
    else if (m_reduction_type == ReductionType::Sum)
    {
        ComputeRawEFieldReduction<amrex::ReduceOpSum>();
    }
}
