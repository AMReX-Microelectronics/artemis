#include "Inductor.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/WarpXUtil.H"
#include "WarpX.H"
#include <ablastr/coarsen/sample.H>
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Scan.H>

#include <AMReX_BaseFwd.H>

#include <memory>
#include <sstream>

Inductor::Inductor ()
{
    amrex::Print() << " Inductor class is constructed \n";
    ReadParameters();
}

void
Inductor::ReadParameters ()
{
    amrex::ParmParse pp_inductor("inductor");

    utils::parser::Store_parserString(pp_inductor, "inductor_x_function(x,y,z)", m_str_inductor_x_function);
    m_inductor_x_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_inductor_x_function, {"x", "y", "z"}));

    utils::parser::Store_parserString(pp_inductor, "inductor_y_function(x,y,z)", m_str_inductor_y_function);
    m_inductor_y_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_inductor_y_function, {"x", "y", "z"}));

    utils::parser::Store_parserString(pp_inductor, "inductor_z_function(x,y,z)", m_str_inductor_z_function);
    m_inductor_z_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_inductor_z_function, {"x", "y", "z"}));
}

#if( AMREX_SPACEDIM == 3)
void
Inductor::InitData()
{
    auto& warpx = WarpX::GetInstance();

    const int lev = 0;
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    // number of guard cells used in EB solver
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();
    // Define a nodal multifab to store if region is on super conductor (1) or not (0)

    amrex::IntVect jx_stag = warpx.get_pointer_current_fp(lev,0)->ixType().toIntVect();
    amrex::IntVect jy_stag = warpx.get_pointer_current_fp(lev,1)->ixType().toIntVect();
    amrex::IntVect jz_stag = warpx.get_pointer_current_fp(lev,2)->ixType().toIntVect();

    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        jx_IndexType[idim] = jx_stag[idim];
        jy_IndexType[idim] = jy_stag[idim];
        jz_IndexType[idim] = jz_stag[idim];
    }

    const int ncomps = 1;
    m_inductor_x_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jx_stag), dmap, ncomps, ng_EB_alloc);
    m_inductor_y_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jy_stag), dmap, ncomps, ng_EB_alloc);
    m_inductor_z_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jz_stag), dmap, ncomps, ng_EB_alloc);

    InitializeInductorMultiFabUsingParser(m_inductor_x_mf.get(), m_inductor_x_parser->compile<3>(), lev);
    InitializeInductorMultiFabUsingParser(m_inductor_y_mf.get(), m_inductor_y_parser->compile<3>(), lev);
    InitializeInductorMultiFabUsingParser(m_inductor_z_mf.get(), m_inductor_z_parser->compile<3>(), lev);
}

void
Inductor::EvolveInductorJ (amrex::Real dt)
{
    amrex::Print() << " evolve inductor J using E\n";
    auto & warpx = WarpX::GetInstance();
    const int lev = 0;

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();

    amrex::MultiFab * jx = warpx.get_pointer_current_fp(lev, 0);
    amrex::MultiFab * jy = warpx.get_pointer_current_fp(lev, 1);
    amrex::MultiFab * jz = warpx.get_pointer_current_fp(lev, 2);

    amrex::MultiFab * Ex = warpx.get_pointer_Efield_fp(lev, 0);
    amrex::MultiFab * Ey = warpx.get_pointer_Efield_fp(lev, 1);
    amrex::MultiFab * Ez = warpx.get_pointer_Efield_fp(lev, 2);

    for (amrex::MFIter mfi(*jx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // Extract pointers to field data
        amrex::Array4<amrex::Real> const& jx_arr = jx->array(mfi);
        amrex::Array4<amrex::Real> const& jy_arr = jy->array(mfi);
        amrex::Array4<amrex::Real> const& jz_arr = jz->array(mfi);
        amrex::Array4<amrex::Real> const& Ex_arr = Ex->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Ey->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez->array(mfi);
        amrex::Array4<amrex::Real> const& inductor_x_arr = m_inductor_x_mf->array(mfi);
        amrex::Array4<amrex::Real> const& inductor_y_arr = m_inductor_y_mf->array(mfi);
        amrex::Array4<amrex::Real> const& inductor_z_arr = m_inductor_z_mf->array(mfi);
        amrex::Box const& tjx = mfi.tilebox(jx->ixType().toIntVect());
        amrex::Box const& tjy = mfi.tilebox(jy->ixType().toIntVect());
        amrex::Box const& tjz = mfi.tilebox(jz->ixType().toIntVect());

        amrex::ParallelFor(tjx, tjy, tjz,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (inductor_x_arr(i,j,k) != 0.) {
                   jx_arr(i,j,k) += dt * Ex_arr(i,j,k) * dx[0] / (dx[1]*dx[2]*inductor_x_arr(i,j,k)); // should be dt * E / (A L')
               }
           },
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (inductor_y_arr(i,j,k) != 0.) {
                   jy_arr(i,j,k) += dt * Ey_arr(i,j,k) * dx[1] / (dx[0]*dx[2]*inductor_y_arr(i,j,k)); // should be dt * E / (A L')
               }
           },
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (inductor_z_arr(i,j,k) != 0.) {
                   jz_arr(i,j,k) += dt * Ez_arr(i,j,k) * dx[2] / (dx[0]*dx[1]*inductor_z_arr(i,j,k)); // should be dt * E / (A L')
               }
           });
    }

}

void
Inductor::InitializeInductorMultiFabUsingParser (amrex::MultiFab *inductor_mf,
                                                 amrex::ParserExecutor<3> const& inductor_parser,
                                                 const int lev)
{
    using namespace amrex::literals;

    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = inductor_mf->ixType().toIntVect();
    for ( amrex::MFIter mfi(*inductor_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const amrex::Box& tb = mfi.tilebox( iv, inductor_mf->nGrowVect());
        amrex::Array4<amrex::Real> const& inductor_fab =  inductor_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type (only 3D supported for now)
                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
                // initialize the macroparameter
                inductor_fab(i,j,k) = inductor_parser(x,y,z);
        });

    }
}


#else
void
Inductor::InitData()
{
    amrex::Abort("Inductor only works with 3D");
}

void
Inductor::EvolveInductorJ (amrex::Real )
{
  amrex::Abort("Inductor only works with 3D");
}
void
Inductor::InitializeInductorMultiFabUsingParser (amrex::MultiFab *,
                                                 amrex::ParserExecutor<3> const& i,
                                                 const int)
{
  amrex::Abort("Inductor only works with 3D");
}
#endif
