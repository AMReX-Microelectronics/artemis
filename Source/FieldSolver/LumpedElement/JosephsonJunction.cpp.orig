#include "JosephsonJunction.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
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

#include <cmath>
#include <memory>
#include <sstream>

JosephsonJunction::JosephsonJunction ()
{
    amrex::Print() << " JosephsonJunction (nonlinear inductor) class is constructed \n";
    ReadParameters();
}

void
JosephsonJunction::ReadParameters ()
{
    amrex::ParmParse pp_jj("josephson");

    utils::parser::Store_parserString(pp_jj, "Ic_x_function(x,y,z)", m_str_Ic_x_function);
    m_Ic_x_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_Ic_x_function, {"x", "y", "z"}));

    utils::parser::Store_parserString(pp_jj, "Ic_y_function(x,y,z)", m_str_Ic_y_function);
    m_Ic_y_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_Ic_y_function, {"x", "y", "z"}));

    utils::parser::Store_parserString(pp_jj, "Ic_z_function(x,y,z)", m_str_Ic_z_function);
    m_Ic_z_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_Ic_z_function, {"x", "y", "z"}));
}

#if( AMREX_SPACEDIM == 3)
void
JosephsonJunction::InitData()
{
    auto& warpx = WarpX::GetInstance();

    const int lev = 0;
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();

    amrex::IntVect jx_stag = warpx.get_pointer_current_fp(lev,0)->ixType().toIntVect();
    amrex::IntVect jy_stag = warpx.get_pointer_current_fp(lev,1)->ixType().toIntVect();
    amrex::IntVect jz_stag = warpx.get_pointer_current_fp(lev,2)->ixType().toIntVect();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        jx_IndexType[idim] = jx_stag[idim];
        jy_IndexType[idim] = jy_stag[idim];
        jz_IndexType[idim] = jz_stag[idim];
    }

    const int ncomps = 1;
    m_Ic_x_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jx_stag), dmap, ncomps, ng_EB_alloc);
    m_Ic_y_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jy_stag), dmap, ncomps, ng_EB_alloc);
    m_Ic_z_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jz_stag), dmap, ncomps, ng_EB_alloc);
    m_phi_x_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jx_stag), dmap, ncomps, ng_EB_alloc);
    m_phi_y_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jy_stag), dmap, ncomps, ng_EB_alloc);
    m_phi_z_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,jz_stag), dmap, ncomps, ng_EB_alloc);

    InitializeJunctionMultiFabUsingParser(m_Ic_x_mf.get(), m_Ic_x_parser->compile<3>(), lev);
    InitializeJunctionMultiFabUsingParser(m_Ic_y_mf.get(), m_Ic_y_parser->compile<3>(), lev);
    InitializeJunctionMultiFabUsingParser(m_Ic_z_mf.get(), m_Ic_z_parser->compile<3>(), lev);

    m_phi_x_mf->setVal(0.0);
    m_phi_y_mf->setVal(0.0);
    m_phi_z_mf->setVal(0.0);
}

void
JosephsonJunction::EvolveJunctionJ (amrex::Real dt)
{
    using namespace amrex::literals;

    amrex::Print() << " evolve Josephson junction: advance phi, add Ic*sin(phi) to J\n";
    auto & warpx = WarpX::GetInstance();
    const int lev = 0;

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();

    // 2 e / hbar = Josephson constant ~ 3.0394e15 rad/(V*s)
    constexpr amrex::Real two_e_over_hbar =
        2.0_rt * PhysConst::q_e / PhysConst::hbar;

    amrex::MultiFab * jx = warpx.get_pointer_current_fp(lev, 0);
    amrex::MultiFab * jy = warpx.get_pointer_current_fp(lev, 1);
    amrex::MultiFab * jz = warpx.get_pointer_current_fp(lev, 2);

    amrex::MultiFab * Ex = warpx.get_pointer_Efield_fp(lev, 0);
    amrex::MultiFab * Ey = warpx.get_pointer_Efield_fp(lev, 1);
    amrex::MultiFab * Ez = warpx.get_pointer_Efield_fp(lev, 2);

    for (amrex::MFIter mfi(*jx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        amrex::Array4<amrex::Real> const& jx_arr = jx->array(mfi);
        amrex::Array4<amrex::Real> const& jy_arr = jy->array(mfi);
        amrex::Array4<amrex::Real> const& jz_arr = jz->array(mfi);
        amrex::Array4<amrex::Real> const& Ex_arr = Ex->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Ey->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez->array(mfi);
        amrex::Array4<amrex::Real> const& Ic_x_arr = m_Ic_x_mf->array(mfi);
        amrex::Array4<amrex::Real> const& Ic_y_arr = m_Ic_y_mf->array(mfi);
        amrex::Array4<amrex::Real> const& Ic_z_arr = m_Ic_z_mf->array(mfi);
        amrex::Array4<amrex::Real> const& phi_x_arr = m_phi_x_mf->array(mfi);
        amrex::Array4<amrex::Real> const& phi_y_arr = m_phi_y_mf->array(mfi);
        amrex::Array4<amrex::Real> const& phi_z_arr = m_phi_z_mf->array(mfi);
        amrex::Box const& tjx = mfi.tilebox(jx->ixType().toIntVect());
        amrex::Box const& tjy = mfi.tilebox(jy->ixType().toIntVect());
        amrex::Box const& tjz = mfi.tilebox(jz->ixType().toIntVect());

        amrex::ParallelFor(tjx, tjy, tjz,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (Ic_x_arr(i,j,k) != 0.) {
                   const amrex::Real V = Ex_arr(i,j,k) * dx[0];
                   phi_x_arr(i,j,k) += dt * two_e_over_hbar * V;
                   const amrex::Real A = dx[1] * dx[2];
                   jx_arr(i,j,k) += Ic_x_arr(i,j,k) * std::sin(phi_x_arr(i,j,k)) / A;
               }
           },
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (Ic_y_arr(i,j,k) != 0.) {
                   const amrex::Real V = Ey_arr(i,j,k) * dx[1];
                   phi_y_arr(i,j,k) += dt * two_e_over_hbar * V;
                   const amrex::Real A = dx[0] * dx[2];
                   jy_arr(i,j,k) += Ic_y_arr(i,j,k) * std::sin(phi_y_arr(i,j,k)) / A;
               }
           },
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               if (Ic_z_arr(i,j,k) != 0.) {
                   const amrex::Real V = Ez_arr(i,j,k) * dx[2];
                   phi_z_arr(i,j,k) += dt * two_e_over_hbar * V;
                   const amrex::Real A = dx[0] * dx[1];
                   jz_arr(i,j,k) += Ic_z_arr(i,j,k) * std::sin(phi_z_arr(i,j,k)) / A;
               }
           });
    }
}

void
JosephsonJunction::InitializeJunctionMultiFabUsingParser (amrex::MultiFab *mf,
                                                          amrex::ParserExecutor<3> const& parser,
                                                          const int lev)
{
    using namespace amrex::literals;

    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = mf->ixType().toIntVect();
    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tb = mfi.tilebox(iv, mf->nGrowVect());
        amrex::Array4<amrex::Real> const& fab = mf->array(mfi);
        amrex::ParallelFor(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
                fab(i,j,k) = parser(x,y,z);
        });
    }
}


#else
void
JosephsonJunction::InitData()
{
    amrex::Abort("JosephsonJunction only works with 3D");
}

void
JosephsonJunction::EvolveJunctionJ (amrex::Real)
{
    amrex::Abort("JosephsonJunction only works with 3D");
}

void
JosephsonJunction::InitializeJunctionMultiFabUsingParser (amrex::MultiFab *,
                                                          amrex::ParserExecutor<3> const&,
                                                          const int)
{
    amrex::Abort("JosephsonJunction only works with 3D");
}
#endif
