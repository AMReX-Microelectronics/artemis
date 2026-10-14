/* Copyright 2019 Andrew Myers, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_WRAPPERS_H_
#define ARTEMIS_WRAPPERS_H_

#include "Particles/ArtemisParticleContainer.H"
#include "Evolve/ArtemisDtType.H"
#include <AMReX_Config.H>
#include <AMReX_REAL.H>

#ifdef AMREX_USE_MPI
#   include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

    int artemis_Real_size();
    int artemis_ParticleReal_size();

    int artemis_nSpecies();

    bool artemis_use_fdtd_nci_corr();

    int artemis_galerkin_interpolation();

    int artemis_nComps();

    int artemis_nCompsSpecies(const char* char_species_name);

    int artemis_SpaceDim();

    void amrex_init (int argc, char* argv[]);

    void amrex_finalize (int finalize_mpi);

    void artemis_init ();

    void artemis_finalize ();

    typedef void(*ARTEMIS_CALLBACK_PY_FUNC_0)();

    void artemis_set_callback_py (const char* char_callback_name,
                                ARTEMIS_CALLBACK_PY_FUNC_0 callback);
    void artemis_clear_callback_py (const char* char_callback_name);

    void artemis_evolve (int numsteps);  // -1 means the inputs parameter will be used.

    void artemis_addNParticles(const char* char_species_name,
                             int lenx,
                             amrex::ParticleReal const * x,
                             amrex::ParticleReal const * y,
                             amrex::ParticleReal const * z,
                             amrex::ParticleReal const * vx,
                             amrex::ParticleReal const * vy,
                             amrex::ParticleReal const * vz,
                             const int nattr_real,
                             amrex::ParticleReal const * attr_real,
                             const int nattr_int,
                             int const * attr_int,
                             int uniqueparticles);

    void artemis_ConvertLabParamsToBoost();

    void artemis_ReadBCParams();

    void artemis_CheckGriddingForRZSpectral();

    amrex::Real artemis_getProbLo(int dir);

    amrex::Real artemis_getProbHi(int dir);

    amrex::Real artemis_getCellSize(int dir, int lev);

    long artemis_getNumParticles(const char* char_species_name, const bool local);

    void artemis_convert_id_to_long (amrex::Long* ids, const ArtemisParticleContainer::ParticleType* pstructs, int size);

    void artemis_convert_cpu_to_int (int* cpus, const ArtemisParticleContainer::ParticleType* pstructs, int size);

    amrex::ParticleReal** artemis_getParticleStructs(
        const char* char_species_name, int lev, int* num_tiles,
        int** particles_per_tile);

    amrex::ParticleReal** artemis_getParticleArrays(
        const char* char_species_name, const char* char_comp_name, int lev,
        int* num_tiles, int** particles_per_tile);

    int artemis_getParticleCompIndex(
        const char* char_species_name, const char* char_comp_name);

    void artemis_addRealComp(
        const char* char_species_name, const char* char_comp_name, bool comm);

    amrex::Real artemis_sumParticleCharge(const char* char_species_name, const bool local);

    int artemis_getParticleBoundaryBufferSize(const char* species_name, int boundary, bool local);

    int** artemis_getParticleBoundaryBufferScrapedSteps(
        const char* species_name, int boundary, int lev,
        int* num_tiles, int** particles_per_tile);

    amrex::ParticleReal** artemis_getParticleBoundaryBuffer(
        const char* species_name, int boundary, int lev,
        int* num_tiles, int** particles_per_tile, const char* comp_name);

    amrex::ParticleReal** artemis_getParticleBoundaryBufferStructs(
            const char* species_name, int boundary, int lev,
            int* num_tiles, int** particles_per_tile);

    void artemis_clearParticleBoundaryBuffer ();

    /**
     * \brief This function is used to deposit a given species' charge density
     * in the rho_fp multifab which can then be accessed from python via
     * pyartemis.fields.RhoFPWrapper()
     *
     * @param[in] species_name specifying the name of the species to deposit
     * @param[in] lev mesh refinement level
     */
    void artemis_depositChargeDensity (const char* species_name, int lev);

  void artemis_ComputeDt ();
  void artemis_MoveWindow (int step, bool move_j);

  void artemis_EvolveE (amrex::Real dt);
  void artemis_EvolveB (amrex::Real dt, DtType a_dt_type);
  void artemis_FillBoundaryE ();
  void artemis_FillBoundaryB ();
  void artemis_SyncRho ();
  void artemis_SyncCurrent (
      const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
      const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp);
  void artemis_UpdateAuxilaryData ();
  void artemis_PushParticlesandDepose (amrex::Real cur_time);

  int artemis_getistep (int lev);
  void artemis_setistep (int lev, int ii);
  amrex::Real artemis_gett_new (int lev);
  void artemis_sett_new (int lev, amrex::Real time);
  amrex::Real artemis_getdt (int lev);

  int artemis_maxStep ();
  amrex::Real artemis_stopTime ();

  int artemis_finestLevel ();

  int artemis_getMyProc ();
  int artemis_getNProcs ();

  void artemis_setPotentialEB (const char * char_potential);

  void mypc_Redistribute ();

  amrex::Real** artemis_getEfield (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getEfieldCP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getEfieldFP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  amrex::Real** artemis_getBfield (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getBfieldCP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getBfieldFP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  amrex::Real** artemis_getCurrentDensity (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getCurrentDensityCP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getCurrentDensityFP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  amrex::Real** artemis_getVectorPotentialFP (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  int* artemis_getEfieldLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getEfieldCPLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getEfieldFPLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  int* artemis_getBfieldLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getBfieldCPLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getBfieldFPLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  int* artemis_getCurrentDensityLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getCurrentDensityCPLoVects (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getCurrentDensityFPLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  int* artemis_getVectorPotentialFPLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getEdgeLengths (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getEdgeLengthsLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getFaceAreas (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getFaceAreasLoVects (int lev, int direction, int *return_size, int **ngrowvect);

  int* artemis_getEx_nodal_flag ();
  int* artemis_getEy_nodal_flag ();
  int* artemis_getEz_nodal_flag ();
  int* artemis_getBx_nodal_flag ();
  int* artemis_getBy_nodal_flag ();
  int* artemis_getBz_nodal_flag ();
  int* artemis_getAx_nodal_flag ();
  int* artemis_getAy_nodal_flag ();
  int* artemis_getAz_nodal_flag ();
  int* artemis_getJx_nodal_flag ();
  int* artemis_getJy_nodal_flag ();
  int* artemis_getJz_nodal_flag ();
  int* artemis_getRho_nodal_flag ();
  int* artemis_getPhi_nodal_flag ();
  int* artemis_getF_nodal_flag ();
  int* artemis_getG_nodal_flag ();
  int* artemis_get_edge_lengths_x_nodal_flag ();
  int* artemis_get_edge_lengths_y_nodal_flag ();
  int* artemis_get_edge_lengths_z_nodal_flag ();
  int* artemis_get_face_areas_x_nodal_flag ();
  int* artemis_get_face_areas_y_nodal_flag ();
  int* artemis_get_face_areas_z_nodal_flag ();

  amrex::Real** artemis_getChargeDensityCP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getChargeDensityFP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getChargeDensityCPLoVects (int lev, int *return_size, int **ngrowvect);
  int* artemis_getChargeDensityFPLoVects (int lev, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getPhiFP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  int* artemis_getPhiFPLoVects (int lev, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getFfieldCP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getFfieldFP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getFfieldCPLoVects (int lev, int *return_size, int **ngrowvect);
  int* artemis_getFfieldFPLoVects (int lev, int *return_size, int **ngrowvect);
  amrex::Real** artemis_getGfieldCP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getGfieldFP (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getGfieldCPLoVects (int lev, int *return_size, int **ngrowvect);
  int* artemis_getGfieldFPLoVects (int lev, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getEfieldCP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getEfieldFP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getBfieldCP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getBfieldFP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getCurrentDensityCP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getCurrentDensityFP_PML (int lev, int direction, int *return_size, int *ncomps, int **ngrowvect, int **shapes);

  int* artemis_getEfieldCPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getEfieldFPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getBfieldCPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getBfieldFPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getCurrentDensityCPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);
  int* artemis_getCurrentDensityFPLoVects_PML (int lev, int direction, int *return_size, int **ngrowvect);

  amrex::Real** artemis_getFfieldCP_PML (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getFfieldFP_PML (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getFfieldCPLoVects_PML (int lev, int *return_size, int **ngrowvect);
  int* artemis_getFfieldFPLoVects_PML (int lev, int *return_size, int **ngrowvect);
  amrex::Real** artemis_getGfieldCP_PML (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  amrex::Real** artemis_getGfieldFP_PML (int lev, int *return_size, int *ncomps, int **ngrowvect, int **shapes);
  int* artemis_getGfieldCPLoVects_PML (int lev, int *return_size, int **ngrowvect);
  int* artemis_getGfieldFPLoVects_PML (int lev, int *return_size, int **ngrowvect);

  int* artemis_getF_pml_nodal_flag ();
  int* artemis_getG_pml_nodal_flag ();

#ifdef __cplusplus
}
#endif

#endif
