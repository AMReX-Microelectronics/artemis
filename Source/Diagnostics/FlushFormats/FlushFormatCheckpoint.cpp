#include "FlushFormatCheckpoint.H"

#include "BoundaryConditions/PML.H"
#if (defined ARTEMIS_DIM_RZ) && (defined ARTEMIS_USE_PSATD)
#   include "BoundaryConditions/PML_RZ.H"
#endif
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Particles/ArtemisParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/ArtemisProfilerWrapper.H"
#include "Artemis.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParticleIO.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

using namespace amrex;

namespace
{
    const std::string default_level_prefix {"Level_"};
}

void
FlushFormatCheckpoint::WriteToFile (
        const amrex::Vector<std::string> /*varnames*/,
        const amrex::Vector<amrex::MultiFab>& /*mf*/,
        amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<int> iteration, const double /*time*/,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
        const std::string prefix, int file_min_digits,
        bool /*plot_raw_fields*/,
        bool /*plot_raw_fields_guards*/,
        const bool /*use_pinned_pc*/,
        bool /*isBTD*/, int /*snapshotID*/,
        int /*bufferID*/, int /*numBuffers*/,
        const amrex::Geometry& /*full_BTD_snapshot*/,
        bool /*isLastBTDFlush*/, const amrex::Vector<int>& /* totalParticlesFlushedAlready*/) const
{
    ARTEMIS_PROFILE("FlushFormatCheckpoint::WriteToFile()");

    auto & artemis = Artemis::GetInstance();

    VisMF::Header::Version current_version = VisMF::GetHeaderVersion();
    VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);

    const std::string& checkpointname = amrex::Concatenate(prefix, iteration[0], file_min_digits);

    amrex::Print() << Utils::TextMsg::Info(
        "Writing checkpoint " + checkpointname);

    // const int nlevels = finestLevel()+1;
    amrex::PreBuildDirectorHierarchy(checkpointname, default_level_prefix, nlev, true);

    WriteArtemisHeader(checkpointname, geom);

    WriteJobInfo(checkpointname);

    for (int lev = 0; lev < nlev; ++lev)
    {
        VisMF::Write(artemis.getEfield_fp(lev, 0),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_fp"));
        VisMF::Write(artemis.getEfield_fp(lev, 1),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_fp"));
        VisMF::Write(artemis.getEfield_fp(lev, 2),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_fp"));
        VisMF::Write(artemis.getBfield_fp(lev, 0),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_fp"));
        VisMF::Write(artemis.getBfield_fp(lev, 1),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_fp"));
        VisMF::Write(artemis.getBfield_fp(lev, 2),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_fp"));

#ifdef ARTEMIS_MAG_LLG
        VisMF::Write(artemis.getHfield_fp(lev, 0),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hx_fp"));
        VisMF::Write(artemis.getHfield_fp(lev, 1),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hy_fp"));
        VisMF::Write(artemis.getHfield_fp(lev, 2),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hz_fp"));
        VisMF::Write(artemis.getMfield_fp(lev, 0),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Mx_fp"));
        VisMF::Write(artemis.getMfield_fp(lev, 1),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "My_fp"));
        VisMF::Write(artemis.getMfield_fp(lev, 2),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Mz_fp"));
        VisMF::Write(artemis.getH_biasfield_fp(lev, 0),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hxbias_fp"));
        VisMF::Write(artemis.getH_biasfield_fp(lev, 1),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hybias_fp"));
        VisMF::Write(artemis.getH_biasfield_fp(lev, 2),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hzbias_fp"));
#endif

        if (Artemis::fft_do_time_averaging)
        {
            VisMF::Write(artemis.getEfield_avg_fp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_avg_fp"));
            VisMF::Write(artemis.getEfield_avg_fp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_avg_fp"));
            VisMF::Write(artemis.getEfield_avg_fp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_avg_fp"));

            VisMF::Write(artemis.getBfield_avg_fp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_avg_fp"));
            VisMF::Write(artemis.getBfield_avg_fp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_avg_fp"));
            VisMF::Write(artemis.getBfield_avg_fp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_avg_fp"));
        }

        if (artemis.getis_synchronized()
            || Artemis::yee_coupled_solver_algo == CoupledYeeSolver::MaxwellLondon
            || artemis.use_lumped_inductor == 1) {
            // Need to save j if synchronized because after restart we need j to evolve E by dt/2.
            VisMF::Write(artemis.getcurrent_fp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jx_fp"));
            VisMF::Write(artemis.getcurrent_fp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jy_fp"));
            VisMF::Write(artemis.getcurrent_fp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jz_fp"));
        }

        if (lev > 0)
        {
            VisMF::Write(artemis.getEfield_cp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_cp"));
            VisMF::Write(artemis.getEfield_cp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_cp"));
            VisMF::Write(artemis.getEfield_cp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_cp"));
            VisMF::Write(artemis.getBfield_cp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_cp"));
            VisMF::Write(artemis.getBfield_cp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_cp"));
            VisMF::Write(artemis.getBfield_cp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_cp"));

#ifdef ARTEMIS_MAG_LLG
            VisMF::Write(artemis.getHfield_cp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hx_cp"));
            VisMF::Write(artemis.getHfield_cp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hy_cp"));
            VisMF::Write(artemis.getHfield_cp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hz_cp"));
            VisMF::Write(artemis.getMfield_cp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Mx_cp"));
            VisMF::Write(artemis.getMfield_cp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "My_cp"));
            VisMF::Write(artemis.getMfield_cp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Mz_cp"));
            VisMF::Write(artemis.getH_biasfield_cp(lev, 0),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hxbias_fp"));
            VisMF::Write(artemis.getH_biasfield_cp(lev, 1),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hybias_fp"));
            VisMF::Write(artemis.getH_biasfield_cp(lev, 2),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Hzbias_fp"));
#endif

            if (Artemis::fft_do_time_averaging)
            {
                VisMF::Write(artemis.getEfield_avg_cp(lev, 0),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_avg_cp"));
                VisMF::Write(artemis.getEfield_avg_cp(lev, 1),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_avg_cp"));
                VisMF::Write(artemis.getEfield_avg_cp(lev, 2),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_avg_cp"));

                VisMF::Write(artemis.getBfield_avg_cp(lev, 0),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_avg_cp"));
                VisMF::Write(artemis.getBfield_avg_cp(lev, 1),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_avg_cp"));
                VisMF::Write(artemis.getBfield_avg_cp(lev, 2),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_avg_cp"));
            }

            if (artemis.getis_synchronized()
                || Artemis::yee_coupled_solver_algo == CoupledYeeSolver::MaxwellLondon
                || artemis.use_lumped_inductor == 1) {
                // Need to save j if synchronized because after restart we need j to evolve E by dt/2.
                VisMF::Write(artemis.getcurrent_cp(lev, 0),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jx_cp"));
                VisMF::Write(artemis.getcurrent_cp(lev, 1),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jy_cp"));
                VisMF::Write(artemis.getcurrent_cp(lev, 2),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jz_cp"));
            }
        }

        if (artemis.DoPML()) {
            if (artemis.GetPML(lev)) {
                artemis.GetPML(lev)->CheckPoint(
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "pml"));
            }
#if (defined ARTEMIS_DIM_RZ) && (defined ARTEMIS_USE_PSATD)
            if (artemis.GetPML_RZ(lev)) {
                artemis.GetPML_RZ(lev)->CheckPoint(
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "pml_rz"));
            }
#endif
        }
    }

    CheckpointParticles(checkpointname, particle_diags);

    WriteDMaps(checkpointname, nlev);

    VisMF::SetHeaderVersion(current_version);

}

void
FlushFormatCheckpoint::CheckpointParticles (
    const std::string& dir,
    const amrex::Vector<ParticleDiag>& particle_diags) const
{
    for (auto& part_diag: particle_diags) {
        ArtemisParticleContainer* pc = part_diag.getParticleContainer();

        Vector<std::string> real_names;
        Vector<std::string> int_names;
        Vector<int> int_flags;
        Vector<int> real_flags;

        real_names.push_back("weight");

        real_names.push_back("momentum_x");
        real_names.push_back("momentum_y");
        real_names.push_back("momentum_z");

#ifdef ARTEMIS_DIM_RZ
        real_names.push_back("theta");
#endif

        // get the names of the real comps
        real_names.resize(pc->NumRealComps());
        auto runtime_rnames = pc->getParticleRuntimeComps();
        for (auto const& x : runtime_rnames) { real_names[x.second+PIdx::nattribs] = x.first; }

        // and the int comps
        int_names.resize(pc->NumIntComps());
        auto runtime_inames = pc->getParticleRuntimeiComps();
        for (auto const& x : runtime_inames) { int_names[x.second+0] = x.first; }

        pc->Checkpoint(dir, part_diag.getSpeciesName(), true,
                       real_names, int_names);
    }
}

void
FlushFormatCheckpoint::WriteDMaps (const std::string& dir, int nlev) const
{
    if (ParallelDescriptor::IOProcessor()) {
        auto & artemis = Artemis::GetInstance();
        for (int lev = 0; lev < nlev; ++lev) {
            std::string DMFileName = dir;
            if (!DMFileName.empty() && DMFileName[DMFileName.size()-1] != '/') {DMFileName += '/';}
            DMFileName = amrex::Concatenate(DMFileName + "Level_", lev, 1);
            DMFileName += "/DM";

            std::ofstream DMFile;
            DMFile.open(DMFileName.c_str(), std::ios::out|std::ios::trunc);

            if (!DMFile.good()) { amrex::FileOpenFailed(DMFileName); }

            DMFile << ParallelDescriptor::NProcs() << "\n";
            artemis.DistributionMap(lev).writeOn(DMFile);

            DMFile.flush();
            DMFile.close();
            ARTEMIS_ALWAYS_ASSERT_WITH_MESSAGE(
                DMFile.good(),
                "FlushFormatCheckpoint::WriteDMaps: problem writing DMFile"
            );
        }
    }
}
