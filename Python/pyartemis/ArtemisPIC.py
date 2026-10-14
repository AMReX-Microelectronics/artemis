# Copyright 2017 David Grote
#
# This file is part of Artemis.
#
# License: BSD-3-Clause-LBNL

# The ArtemisPIC class is the beginnings of an implementation of a standard interface
# for running PIC codes. This is the run time equivalent to the PICMI standard.
# This standard would specify the API for calling the various pieces of typical
# time step loops, for example get_self_fields and put_Efields. Ideally, a user
# could write a loop using the standard and, importing one of compliant codes, be
# able to run a customized PIC simulation with that code.

from warp.run_modes.timestepper import PICAPI

from ._libartemis import libartemis


class ArtemisPIC(PICAPI):

    def get_time(self):
        return libartemis.libartemis_so.artemis_gett_new(0)

    def set_time(self, time):
        for i in range(libartemis.libartemis_so.artemis_finestLevel()+1):
            libartemis.libartemis_so.artemis_sett_new(i, time)

    def get_step_size(self):
        libartemis.libartemis_so.artemis_ComputeDt()
        return libartemis.libartemis_so.artemis_getdt(0)

    def get_step_number(self):
        return libartemis.libartemis_so.artemis_getistep(0)

    def set_step_number(self, it):
        for i in range(libartemis.libartemis_so.artemis_finestLevel()+1):
            libartemis.libartemis_so.artemis_setistep(i, it)

    def push_positions(self, dt):
        libartemis.libartemis_so.artemis_PushX(0, dt)

    def push_velocities_withE(self, dt):
        libartemis.libartemis_so.artemis_EPushV(0, dt)

    def push_velocities_withB(self, dt):
        libartemis.libartemis_so.artemis_BPushV(0, dt)

    def get_self_fields(self):
        libartemis.libartemis_so.artemis_FieldGather(0)

    def calculate_source(self):
        libartemis.libartemis_so.artemis_CurrentDeposition(0)

    def push_Efields(self, dt):
        libartemis.libartemis_so.artemis_EvolveE(0, dt)
        libartemis.libartemis_so.artemis_FillBoundaryE(0, True)

    def push_Bfields(self, dt):
        libartemis.libartemis_so.artemis_EvolveB(0, dt)
        libartemis.libartemis_so.artemis_FillBoundaryB(0, True)

    def apply_particle_boundary_conditions(self):
        libartemis.libartemis_so.mypc_Redistribute() # Redistribute particles
        libartemis.libartemis_so.artemis_MoveWindow(self.istep,True) # !!! not the correct place yet
