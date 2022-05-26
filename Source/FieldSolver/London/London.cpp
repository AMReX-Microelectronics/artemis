#include "London.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/WarpXUtil.H"
#include "Utils/CoarsenIO.H"
#include "WarpX.H"
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

London::London ()
{
    amrex::Print() << " London class is constructed\n";
    ReadParameters();
}

void
London::ReadParameters ()
{
    amrex::ParmParse pp_london("london");
    pp_london.get("penetration_depth", m_penetration_depth);

    Store_parserString(pp_london, "superconductor_function(x,y,z)", m_str_superconductor_function);
    m_superconductor_parser = std::make_unique<amrex::Parser>(
                                   makeParser(m_str_superconductor_function, {"x", "y", "z"}));
}

void
London::InitData()
{
    auto& warpx = WarpX::GetInstance();

    const int lev = 0;
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    // number of guard cells used in EB solver
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();
    // Define a nodal multifab to store if region is on super conductor (1) or not (0)
    const amrex::IntVect nodal_flag = amrex::IntVect::TheNodeVector();
    const int ncomps = 1;
    m_superconductor_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,nodal_flag), dmap, ncomps, ng_EB_alloc);

    InitializeSuperconductorMultiFabUsingParser(m_superconductor_mf.get(), m_superconductor_parser->compile<3>(), lev);

    amrex::IntVect jx_stag = warpx.get_pointer_current_fp(lev,0)->ixType().toIntVect();
    amrex::IntVect jy_stag = warpx.get_pointer_current_fp(lev,1)->ixType().toIntVect();
    amrex::IntVect jz_stag = warpx.get_pointer_current_fp(lev,2)->ixType().toIntVect();

    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        jx_IndexType[idim]      = jx_stag[idim];
        jy_IndexType[idim]      = jy_stag[idim];
        jz_IndexType[idim]      = jz_stag[idim];
    }

}

void
London::EvolveLondonJ (amrex::Real dt)
{
    amrex::Print() << " evolve london J using E\n";
    auto & warpx = WarpX::GetInstance();
    const int lev = 0;

    amrex::MultiFab * jx = warpx.get_pointer_current_fp(lev, 0);
    amrex::MultiFab * jy = warpx.get_pointer_current_fp(lev, 1);
    amrex::MultiFab * jz = warpx.get_pointer_current_fp(lev, 2);

    amrex::MultiFab * Ex = warpx.get_pointer_Efield_fp(lev, 0);
    amrex::MultiFab * Ey = warpx.get_pointer_Efield_fp(lev, 1);
    amrex::MultiFab * Ez = warpx.get_pointer_Efield_fp(lev, 2);

    MacroscopicProperties &macroscopic = warpx.GetMacroscopicProperties();
    amrex::MultiFab& mu_mf = macroscopic.getmu_mf();
    amrex::GpuArray<int, 3> const& mu_stag = macroscopic.mu_IndexType;
    amrex::GpuArray<int, 3> const& jx_stag = jx_IndexType;
    amrex::GpuArray<int, 3> const& jy_stag = jy_IndexType;
    amrex::GpuArray<int, 3> const& jz_stag = jz_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr     = macroscopic.macro_cr_ratio;

    // evolve J  = 1/( (lambda*lambda) * mu) * E * dt

    amrex::Real lambda_sq_inv = 1.0/(m_penetration_depth*m_penetration_depth);
    const int scomp = 0;
    for (amrex::MFIter mfi(*jx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        //Extract field data
        amrex::Array4<amrex::Real> const& jx_arr = jx->array(mfi);
        amrex::Array4<amrex::Real> const& jy_arr = jy->array(mfi);
        amrex::Array4<amrex::Real> const& jz_arr = jz->array(mfi);
        amrex::Array4<amrex::Real> const& Ex_arr = Ex->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Ey->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez->array(mfi);
        amrex::Array4<amrex::Real> const& mu_arr = mu_mf.array(mfi);
        amrex::Array4<amrex::Real> const& sc_arr = m_superconductor_mf->array(mfi);
        amrex::Box const& tjx = mfi.tilebox(jx->ixType().toIntVect());
        amrex::Box const& tjy = mfi.tilebox(jy->ixType().toIntVect());
        amrex::Box const& tjz = mfi.tilebox(jz->ixType().toIntVect());


    amrex::ParallelFor(tjx, tjy, tjz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (sc_arr(i,j,k)==1 and sc_arr(i+1,j,k)==1) {
                amrex::Real const mu_interp = CoarsenIO::Interp(mu_arr, mu_stag, jx_stag,
                                                                macro_cr, i, j, k, scomp);
                jx_arr(i,j,k) += dt * lambda_sq_inv/mu_interp * Ex_arr(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (sc_arr(i,j,k)==1 and sc_arr(i,j+1,k)==1) {
                amrex::Real const mu_interp = CoarsenIO::Interp(mu_arr, mu_stag, jy_stag,
                                                                macro_cr, i, j, k, scomp);
                jy_arr(i,j,k) += dt * lambda_sq_inv/mu_interp * Ey_arr(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (sc_arr(i,j,k)==1 and sc_arr(i,j,k+1)==1) {
                amrex::Real const mu_interp = CoarsenIO::Interp(mu_arr, mu_stag, jz_stag,
                                                                macro_cr, i, j, k, scomp);
                jz_arr(i,j,k) += dt * lambda_sq_inv/mu_interp * Ez_arr(i,j,k);
            }
        }
    );
    }

}

void
London::InitializeSuperconductorMultiFabUsingParser (
                       amrex::MultiFab *sc_mf,
                       amrex::ParserExecutor<3> const& sc_parser,
                       const int lev)
{
    using namespace amrex::literals;

    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = sc_mf->ixType().toIntVect();
    for ( amrex::MFIter mfi(*sc_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const amrex::Box& tb = mfi.tilebox( iv, sc_mf->nGrowVect());
        amrex::Array4<amrex::Real> const& sc_fab =  sc_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type (only 3D supported for now)
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
                // initialize the macroparameter
                sc_fab(i,j,k) = sc_parser(x,y,z);
        });

    }
}

