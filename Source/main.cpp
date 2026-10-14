/* Copyright 2016-2020 Andrew Myers, Ann Almgren, Axel Huebl
 *                     David Grote, Jean-Luc Vay, Remi Lehe
 *                     Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Artemis.H"

#include "Initialization/ArtemisAMReXInit.H"
#include "Utils/MPIInitHelpers.H"
#include "Utils/ArtemisProfilerWrapper.H"
#include "Utils/ArtemisrocfftUtil.H"
#include "Utils/ArtemisUtil.H"

#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/utils/timer/Timer.H>

#include <AMReX_Print.H>

int main(int argc, char* argv[])
{
    utils::artemis_mpi_init(argc, argv);

    artemis_amrex_init(argc, argv);

    utils::rocfft::setup();

    ParseGeometryInput();

    ConvertLabParamsToBoost();
    ReadBCParams();

#ifdef ARTEMIS_DIM_RZ
    CheckGriddingForRZSpectral();
#endif

    {
        ARTEMIS_PROFILE_VAR("main()", pmain);

        auto timer = ablastr::utils::timer::Timer{};
        timer.record_start_time();

        Artemis artemis;

        artemis.InitData();

        artemis.Evolve();

        //Print warning messages at the end of the simulation
        ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("THE END");

        timer.record_stop_time();
        if (artemis.Verbose()) {
            amrex::Print() << "Total Time                     : "
                    << timer.get_global_duration() << '\n';
        }

        ARTEMIS_PROFILE_VAR_STOP(pmain);
    }

    utils::rocfft::cleanup();

    amrex::Finalize();

    utils::artemis_mpi_finalize ();
}
