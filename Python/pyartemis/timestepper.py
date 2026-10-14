# Copyright 2017-2021 Andrew Myers, David Grote, Weiqun Zhang
#
#
# This file is part of Artemis.
#
# License: BSD-3-Clause-LBNL

# This is intended to be a Python level example of how one might write a customized
# time stepping loop. This would replace the functionality of the Artemis::Evolve routine.
# Wrappers are available for the major pieces of a time step and they would be called
# here in the appropriate order.
# Note that this is intended to be an example only and may not be functional. The
# onestep routine as written here is out of date and is not consistent with Artemis::Evolve.

from . import callbacks, libartemis


class TimeStepper(object):

    def step(self, nsteps=1):
        for i in range(nsteps):
            self.onestep()

    def onestep(self):

        callbacks._beforestep()

        self.cur_time = libartemis.libartemis_so.artemis_gett_new(0)
        self.istep = libartemis.libartemis_so.artemis_getistep(0)

        #if mpi.rank == 0:
        print("\nSTEP %d starts ..."%(self.istep + 1))

        #if (ParallelDescriptor::NProcs() > 1)
        #   if (okToRegrid(step)) RegridBaseLevel();

        dt = libartemis.libartemis_so.artemis_getdt(0)

        # --- At the beginning, we have B^{n-1/2} and E^{n}.
        # --- Particles have p^{n-1/2} and x^{n}.
        libartemis.libartemis_so.artemis_FillBoundaryE()
        libartemis.libartemis_so.artemis_EvolveB(0.5*dt,1) # We now B^{n}

        libartemis.libartemis_so.artemis_FillBoundaryB()
        libartemis.libartemis_so.artemis_UpdateAuxilaryData()

        # --- Evolve particles to p^{n+1/2} and x^{n+1}
        # --- Depose current, j^{n+1/2}
        callbacks._particleinjection()
        callbacks._particlescraper()
        callbacks._beforedeposition()
        libartemis.libartemis_so.artemis_PushParticlesandDepose(self.cur_time)
        callbacks._afterdeposition()

        libartemis.libartemis_so.mypc_Redistribute() # Redistribute particles

        libartemis.libartemis_so.artemis_FillBoundaryE()
        libartemis.libartemis_so.artemis_EvolveB(0.5*dt,2) # We now B^{n+1/2}

        libartemis.libartemis_so.artemis_SyncCurrent()

        libartemis.libartemis_so.artemis_FillBoundaryB()
        callbacks._beforeEsolve()
        libartemis.libartemis_so.artemis_EvolveE(dt,0) # We now have E^{n+1}
        callbacks._afterEsolve()

        self.istep += 1

        self.cur_time += dt

        libartemis.libartemis_so.artemis_MoveWindow(self.istep,True);

        #if mpi.rank == 0:
        print("STEP %d ends. TIME = %e DT = %e"%(self.istep, self.cur_time, dt))

        # --- Sync up time
        for i in range(libartemis.libartemis_so.artemis_finestLevel()+1):
            libartemis.libartemis_so.artemis_sett_new(i, self.cur_time)
            libartemis.libartemis_so.artemis_setistep(i, self.istep)

        callbacks._afterstep()
