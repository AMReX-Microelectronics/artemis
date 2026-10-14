/* Copyright 2019 Andrew Myers, Axel Huebl, David Grote
 * Luca Fedeli, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BoundaryConditions/PML.H"
#include "Initialization/ArtemisAMReXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/ArtemisParticleContainer.H"
#include "Utils/ArtemisProfilerWrapper.H"
#include "Utils/ArtemisUtil.H"
#include "Artemis.H"
#include "ArtemisWrappers.H"
#include "Artemis_py.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>
#include <AMReX_StructOfArrays.H>

#include <array>
#include <cstdlib>

namespace
{
    amrex::Real** getMultiFabPointers (amrex::MultiFab& mf, int *num_boxes, int *ncomps, int **ngrowvect, int **shapes)
    {
        *ncomps = mf.nComp();
        *num_boxes = mf.local_size();
        int shapesize = AMREX_SPACEDIM;
        *ngrowvect = static_cast<int*>(malloc(sizeof(int)*shapesize));
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            (*ngrowvect)[j] = mf.nGrow(j);
        }
        if (mf.nComp() > 1) shapesize += 1;
        *shapes = static_cast<int*>(malloc(sizeof(int)*shapesize * (*num_boxes)));
        auto data =
            static_cast<amrex::Real**>(malloc((*num_boxes) * sizeof(amrex::Real*)));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi ) {
            int i = mfi.LocalIndex();
            data[i] = mf[mfi].dataPtr();
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                (*shapes)[shapesize*i+j] = mf[mfi].box().length(j);
            }
            if (mf.nComp() > 1) (*shapes)[shapesize*i+AMREX_SPACEDIM] = mf.nComp();
        }
        return data;
    }
    int* getMultiFabLoVects (const amrex::MultiFab& mf, int *num_boxes, int **ngrowvect)
    {
        int shapesize = AMREX_SPACEDIM;
        *ngrowvect = static_cast<int*>(malloc(sizeof(int)*shapesize));
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            (*ngrowvect)[j] = mf.nGrow(j);
        }
        *num_boxes = mf.local_size();
        auto loVects = static_cast<int*>(malloc((*num_boxes)*AMREX_SPACEDIM * sizeof(int)));

        int i = 0;
        for ( amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi, ++i ) {
            const int* loVect = mf[mfi].loVect();
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                loVects[AMREX_SPACEDIM*i+j] = loVect[j];
            }
        }
        return loVects;
    }
    // Copy the nodal flag data and return the copy:
    // the nodal flag data should not be modifiable from Python.
    int* getFieldNodalFlagData ( const amrex::MultiFab* mf )
    {
        if (mf == nullptr) return nullptr;
        const amrex::IntVect nodal_flag( mf->ixType().toIntVect() );
        auto *nodal_flag_data = static_cast<int*>(malloc(AMREX_SPACEDIM * sizeof(int)));

        constexpr int NODE = amrex::IndexType::NODE;

        for (int i=0 ; i < AMREX_SPACEDIM ; i++) {
            nodal_flag_data[i] = (nodal_flag[i] == NODE ? 1 : 0);
        }
        return nodal_flag_data;
    }
}

    int artemis_Real_size()
    {
        return (int)sizeof(amrex::Real);
    }

    int artemis_ParticleReal_size()
    {
        return (int)sizeof(amrex::ParticleReal);
    }

    int artemis_nSpecies()
    {
        const auto & mypc = Artemis::GetInstance().GetPartContainer();
        return mypc.nSpecies();
    }

    bool artemis_use_fdtd_nci_corr()
    {
        return Artemis::use_fdtd_nci_corr;
    }

    int artemis_galerkin_interpolation()
    {
        return Artemis::galerkin_interpolation;
    }

    int artemis_nComps()
    {
        return PIdx::nattribs;
    }

    int artemis_nCompsSpecies(const char* char_species_name)
    {
        auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);
        return myspc.NumRealComps();
    }

    int artemis_SpaceDim()
    {
        return AMREX_SPACEDIM;
    }

    void amrex_init (int argc, char* argv[])
    {
        artemis_amrex_init(argc, argv);
    }

    void amrex_finalize (int /*finalize_mpi*/)
    {
        amrex::Finalize();
    }

    void artemis_init ()
    {
        Artemis& artemis = Artemis::GetInstance();
        artemis.InitData();
        ExecutePythonCallback("afterinit");
        ExecutePythonCallback("particleloader");
    }

    void artemis_finalize ()
    {
        Artemis::ResetInstance();
    }

    void artemis_set_callback_py (
        const char* char_callback_name, ARTEMIS_CALLBACK_PY_FUNC_0 callback)
    {
        const std::string callback_name(char_callback_name);
        artemis_callback_py_map[callback_name] = callback;
    }

    void artemis_clear_callback_py (const char* char_callback_name)
    {
        const std::string callback_name(char_callback_name);
        artemis_callback_py_map.erase(callback_name);
    }

    void artemis_evolve (int numsteps)
    {
        Artemis& artemis = Artemis::GetInstance();
        artemis.Evolve(numsteps);
    }

    void artemis_addNParticles(
        const char* char_species_name, int lenx, amrex::ParticleReal const * x,
        amrex::ParticleReal const * y, amrex::ParticleReal const * z,
        amrex::ParticleReal const * vx, amrex::ParticleReal const * vy,
        amrex::ParticleReal const * vz, const int nattr_real,
        amrex::ParticleReal const * attr_real, const int nattr_int,
        int const * attr_int, int uniqueparticles)
    {
        auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);
        const int lev = 0;
        myspc.AddNParticles(lev, lenx, x, y, z, vx, vy, vz, nattr_real, attr_real,
                            nattr_int, attr_int, uniqueparticles);
    }

    void artemis_ConvertLabParamsToBoost()
    {
      ConvertLabParamsToBoost();
    }

    void artemis_ReadBCParams()
    {
      ReadBCParams();
    }

    void artemis_CheckGriddingForRZSpectral()
    {
        CheckGriddingForRZSpectral();
    }

    amrex::Real artemis_getProbLo(int dir)
    {
      Artemis& artemis = Artemis::GetInstance();
      const amrex::Geometry& geom = artemis.Geom(0);
      return geom.ProbLo(dir);
    }

    amrex::Real artemis_getProbHi(int dir)
    {
      Artemis& artemis = Artemis::GetInstance();
      const amrex::Geometry& geom = artemis.Geom(0);
      return geom.ProbHi(dir);
    }

    amrex::Real artemis_getCellSize(int dir, int lev) {
        const std::array<amrex::Real,3>& dx = Artemis::CellSize(lev);
        return dx[dir];
    }

    long artemis_getNumParticles(const char* char_species_name, const bool local) {
        const auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);
        // the first argument below is to only count valid particles
        return myspc.TotalNumberOfParticles(true, local);
    }

#define ARTEMIS_GET_FIELD(FIELD, GETTER) \
    amrex::Real** FIELD(int lev, int direction, \
                        int *return_size, int *ncomps, int **ngrowvect, int **shapes) { \
        auto * mf = GETTER(lev, direction); \
        if (mf != nullptr) { \
            return getMultiFabPointers(*mf, return_size, ncomps, ngrowvect, shapes); \
        } else { \
            return nullptr; \
        } \
    }

#define ARTEMIS_GET_LOVECTS(FIELD, GETTER) \
    int* FIELD(int lev, int direction, \
               int *return_size, int **ngrowvect) { \
        auto * mf = GETTER(lev, direction); \
        if (mf != nullptr) { \
            return getMultiFabLoVects(*mf, return_size, ngrowvect); \
        } else { \
            return nullptr; \
        } \
    }

    ARTEMIS_GET_FIELD(artemis_getEfield, Artemis::GetInstance().get_pointer_Efield_aux)
    ARTEMIS_GET_FIELD(artemis_getEfieldCP, Artemis::GetInstance().get_pointer_Efield_cp)
    ARTEMIS_GET_FIELD(artemis_getEfieldFP, Artemis::GetInstance().get_pointer_Efield_fp)

    ARTEMIS_GET_FIELD(artemis_getBfield, Artemis::GetInstance().get_pointer_Bfield_aux)
    ARTEMIS_GET_FIELD(artemis_getBfieldCP, Artemis::GetInstance().get_pointer_Bfield_cp)
    ARTEMIS_GET_FIELD(artemis_getBfieldFP, Artemis::GetInstance().get_pointer_Bfield_fp)

    ARTEMIS_GET_FIELD(artemis_getEdgeLengths, Artemis::GetInstance().get_pointer_edge_lengths)
    ARTEMIS_GET_FIELD(artemis_getFaceAreas, Artemis::GetInstance().get_pointer_face_areas)

    ARTEMIS_GET_FIELD(artemis_getCurrentDensity, Artemis::GetInstance().get_pointer_current_fp)
    ARTEMIS_GET_FIELD(artemis_getCurrentDensityCP, Artemis::GetInstance().get_pointer_current_cp)
    ARTEMIS_GET_FIELD(artemis_getCurrentDensityFP, Artemis::GetInstance().get_pointer_current_fp)

    ARTEMIS_GET_FIELD(artemis_getVectorPotentialFP, Artemis::GetInstance().get_pointer_vector_potential_fp)

    ARTEMIS_GET_LOVECTS(artemis_getEfieldLoVects, Artemis::GetInstance().get_pointer_Efield_aux)
    ARTEMIS_GET_LOVECTS(artemis_getEfieldCPLoVects, Artemis::GetInstance().get_pointer_Efield_cp)
    ARTEMIS_GET_LOVECTS(artemis_getEfieldFPLoVects, Artemis::GetInstance().get_pointer_Efield_fp)

    ARTEMIS_GET_LOVECTS(artemis_getBfieldLoVects, Artemis::GetInstance().get_pointer_Bfield_aux)
    ARTEMIS_GET_LOVECTS(artemis_getBfieldCPLoVects, Artemis::GetInstance().get_pointer_Bfield_cp)
    ARTEMIS_GET_LOVECTS(artemis_getBfieldFPLoVects, Artemis::GetInstance().get_pointer_Bfield_fp)

    ARTEMIS_GET_LOVECTS(artemis_getCurrentDensityLoVects, Artemis::GetInstance().get_pointer_current_fp)
    ARTEMIS_GET_LOVECTS(artemis_getCurrentDensityCPLoVects, Artemis::GetInstance().get_pointer_current_cp)
    ARTEMIS_GET_LOVECTS(artemis_getCurrentDensityFPLoVects, Artemis::GetInstance().get_pointer_current_fp)
    ARTEMIS_GET_LOVECTS(artemis_getVectorPotentialFPLoVects, Artemis::GetInstance().get_pointer_vector_potential_fp)

    ARTEMIS_GET_LOVECTS(artemis_getEdgeLengthsLoVects, Artemis::GetInstance().get_pointer_edge_lengths)
    ARTEMIS_GET_LOVECTS(artemis_getFaceAreasLoVects, Artemis::GetInstance().get_pointer_face_areas)

    int* artemis_getEx_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Efield_aux(0,0) );}
    int* artemis_getEy_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Efield_aux(0,1) );}
    int* artemis_getEz_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Efield_aux(0,2) );}
    int* artemis_getBx_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Bfield_aux(0,0) );}
    int* artemis_getBy_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Bfield_aux(0,1) );}
    int* artemis_getBz_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_Bfield_aux(0,2) );}
    int* artemis_getJx_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_current_fp(0,0) );}
    int* artemis_getJy_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_current_fp(0,1) );}
    int* artemis_getJz_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_current_fp(0,2) );}
    int* artemis_getAx_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_vector_potential_fp(0,0) );}
    int* artemis_getAy_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_vector_potential_fp(0,1) );}
    int* artemis_getAz_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_vector_potential_fp(0,2) );}
    int* artemis_getRho_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_rho_fp(0) );}
    int* artemis_getPhi_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_phi_fp(0) );}
    int* artemis_getF_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_F_fp(0) );}
    int* artemis_getG_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_G_fp(0) );}
    int* artemis_get_edge_lengths_x_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_edge_lengths(0, 0) );}
    int* artemis_get_edge_lengths_y_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_edge_lengths(0, 1) );}
    int* artemis_get_edge_lengths_z_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_edge_lengths(0, 2) );}
    int* artemis_get_face_areas_x_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_face_areas(0, 0) );}
    int* artemis_get_face_areas_y_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_face_areas(0, 1) );}
    int* artemis_get_face_areas_z_nodal_flag() {return getFieldNodalFlagData( Artemis::GetInstance().get_pointer_face_areas(0, 2) );}

#define ARTEMIS_GET_SCALAR(SCALAR, GETTER) \
    amrex::Real** SCALAR(int lev, \
                         int *return_size, int *ncomps, int **ngrowvect, int **shapes) { \
        auto * mf = GETTER(lev); \
        if (mf != nullptr) { \
            return getMultiFabPointers(*mf, return_size, ncomps, ngrowvect, shapes); \
        } else { \
            return nullptr; \
        } \
    }

#define ARTEMIS_GET_LOVECTS_SCALAR(SCALAR, GETTER) \
    int* SCALAR(int lev, \
                int *return_size, int **ngrowvect) { \
        auto * mf = GETTER(lev); \
        if (mf != nullptr) { \
            return getMultiFabLoVects(*mf, return_size, ngrowvect); \
        } else { \
            return nullptr; \
        } \
    }

    ARTEMIS_GET_SCALAR(artemis_getChargeDensityCP, Artemis::GetInstance().get_pointer_rho_cp)
    ARTEMIS_GET_SCALAR(artemis_getChargeDensityFP, Artemis::GetInstance().get_pointer_rho_fp)

    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getChargeDensityCPLoVects, Artemis::GetInstance().get_pointer_rho_cp)
    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getChargeDensityFPLoVects, Artemis::GetInstance().get_pointer_rho_fp)

    ARTEMIS_GET_SCALAR(artemis_getPhiFP, Artemis::GetInstance().get_pointer_phi_fp)

    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getPhiFPLoVects, Artemis::GetInstance().get_pointer_phi_fp)

    // F and G
    ARTEMIS_GET_SCALAR(artemis_getFfieldCP, Artemis::GetInstance().get_pointer_F_cp)
    ARTEMIS_GET_SCALAR(artemis_getFfieldFP, Artemis::GetInstance().get_pointer_F_fp)
    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getFfieldCPLoVects, Artemis::GetInstance().get_pointer_F_cp)
    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getFfieldFPLoVects, Artemis::GetInstance().get_pointer_F_fp)
    ARTEMIS_GET_SCALAR(artemis_getGfieldCP, Artemis::GetInstance().get_pointer_G_cp)
    ARTEMIS_GET_SCALAR(artemis_getGfieldFP, Artemis::GetInstance().get_pointer_G_fp)
    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getGfieldCPLoVects, Artemis::GetInstance().get_pointer_G_cp)
    ARTEMIS_GET_LOVECTS_SCALAR(artemis_getGfieldFPLoVects, Artemis::GetInstance().get_pointer_G_fp)

#define ARTEMIS_GET_FIELD_PML(FIELD, GETTER) \
    amrex::Real** FIELD(int lev, int direction, \
                        int *return_size, int *ncomps, int **ngrowvect, int **shapes) { \
        auto * pml = Artemis::GetInstance().GetPML(lev); \
        if (!pml) return nullptr; \
        auto * mf = (pml->GETTER()[direction]); \
        if (!mf) return nullptr; \
        return getMultiFabPointers(*mf, return_size, ncomps, ngrowvect, shapes); \
    }

#define ARTEMIS_GET_LOVECTS_PML(FIELD, GETTER) \
    int* FIELD(int lev, int direction, \
               int *return_size, int **ngrowvect) { \
        auto * pml = Artemis::GetInstance().GetPML(lev); \
        if (!pml) return nullptr; \
        auto * mf = (pml->GETTER()[direction]); \
        if (!mf) return nullptr; \
        return getMultiFabLoVects(*mf, return_size, ngrowvect); \
    }

    ARTEMIS_GET_FIELD_PML(artemis_getEfieldCP_PML, GetE_cp)
    ARTEMIS_GET_FIELD_PML(artemis_getEfieldFP_PML, GetE_fp)
    ARTEMIS_GET_FIELD_PML(artemis_getBfieldCP_PML, GetB_cp)
    ARTEMIS_GET_FIELD_PML(artemis_getBfieldFP_PML, GetB_fp)
    ARTEMIS_GET_FIELD_PML(artemis_getCurrentDensityCP_PML, Getj_cp)
    ARTEMIS_GET_FIELD_PML(artemis_getCurrentDensityFP_PML, Getj_fp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getEfieldCPLoVects_PML, GetE_cp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getEfieldFPLoVects_PML, GetE_fp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getBfieldCPLoVects_PML, GetB_cp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getBfieldFPLoVects_PML, GetB_fp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getCurrentDensityCPLoVects_PML, Getj_cp)
    ARTEMIS_GET_LOVECTS_PML(artemis_getCurrentDensityFPLoVects_PML, Getj_fp)

#define ARTEMIS_GET_SCALAR_PML(SCALAR, GETTER) \
    amrex::Real** SCALAR(int lev, \
                        int *return_size, int *ncomps, int **ngrowvect, int **shapes) { \
        auto * pml = Artemis::GetInstance().GetPML(lev); \
        if (!pml) return nullptr; \
        auto * mf = pml->GETTER(); \
        if (!mf) return nullptr; \
        return getMultiFabPointers(*mf, return_size, ncomps, ngrowvect, shapes); \
    }

#define ARTEMIS_GET_LOVECTS_PML_SCALAR(SCALAR, GETTER) \
    int* SCALAR(int lev, \
               int *return_size, int **ngrowvect) { \
        auto * pml = Artemis::GetInstance().GetPML(lev); \
        if (!pml) return nullptr; \
        auto * mf = pml->GETTER(); \
        if (!mf) return nullptr; \
        return getMultiFabLoVects(*mf, return_size, ngrowvect); \
    }

    // F and G
    ARTEMIS_GET_SCALAR_PML(artemis_getFfieldCP_PML, GetF_cp)
    ARTEMIS_GET_SCALAR_PML(artemis_getFfieldFP_PML, GetF_fp)
    ARTEMIS_GET_LOVECTS_PML_SCALAR(artemis_getFfieldCPLoVects_PML, GetF_cp)
    ARTEMIS_GET_LOVECTS_PML_SCALAR(artemis_getFfieldFPLoVects_PML, GetF_fp)
    ARTEMIS_GET_SCALAR_PML(artemis_getGfieldCP_PML, GetG_cp)
    ARTEMIS_GET_SCALAR_PML(artemis_getGfieldFP_PML, GetG_fp)
    ARTEMIS_GET_LOVECTS_PML_SCALAR(artemis_getGfieldCPLoVects_PML, GetG_cp)
    ARTEMIS_GET_LOVECTS_PML_SCALAR(artemis_getGfieldFPLoVects_PML, GetG_fp)

    int* artemis_getF_pml_nodal_flag ()
    {
        auto * pml = Artemis::GetInstance().GetPML(0);
        if (!pml) return nullptr;
        return getFieldNodalFlagData(pml->GetF_fp());
    }

    int* artemis_getG_pml_nodal_flag ()
    {
        auto * pml = Artemis::GetInstance().GetPML(0);
        if (!pml) return nullptr;
        return getFieldNodalFlagData(pml->GetG_fp());
    }

    amrex::ParticleReal** artemis_getParticleStructs(
            const char* char_species_name, int lev,
            int* num_tiles, int** particles_per_tile) {
        const auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);

        *num_tiles = myspc.numLocalTilesAtLevel(lev);
        *particles_per_tile = static_cast<int*>(malloc(*num_tiles*sizeof(int)));
        memset(*particles_per_tile, 0, *num_tiles*sizeof(int));

        auto data = static_cast<amrex::ParticleReal**>(malloc(*num_tiles*sizeof(typename ArtemisParticleContainer::ParticleType*)));
        int i = 0;
        for (ArtemisParIter pti(myspc, lev); pti.isValid(); ++pti, ++i) {
            auto& aos = pti.GetArrayOfStructs();
            data[i] = (amrex::ParticleReal*) aos.data();
            (*particles_per_tile)[i] = pti.numParticles();
        }
        return data;
    }

    amrex::ParticleReal** artemis_getParticleArrays (
            const char* char_species_name, const char* char_comp_name,
            int lev, int* num_tiles, int** particles_per_tile ) {

        const auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);

        int comp = artemis_getParticleCompIndex(char_species_name, char_comp_name);

        *num_tiles = myspc.numLocalTilesAtLevel(lev);
        *particles_per_tile = static_cast<int*>(malloc(*num_tiles*sizeof(int)));
        memset(*particles_per_tile, 0, *num_tiles*sizeof(int));

        auto data = static_cast<amrex::ParticleReal**>(malloc(*num_tiles*sizeof(amrex::ParticleReal*)));
        int i = 0;
        for (ArtemisParIter pti(myspc, lev); pti.isValid(); ++pti, ++i) {
            auto& soa = pti.GetStructOfArrays();
            data[i] = (amrex::ParticleReal*) soa.GetRealData(comp).dataPtr();
            (*particles_per_tile)[i] = pti.numParticles();
        }
        return data;
    }

    void artemis_convert_id_to_long (amrex::Long* ids, const ArtemisParticleContainer::ParticleType* pstructs, int size)
    {
        amrex::Long* d_ptr = nullptr;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Long> d_ids(size);
        d_ptr = d_ids.data();
#else
        d_ptr = ids;
#endif
        amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            d_ptr[i] = pstructs[i].id();
        });
#ifdef AMREX_USE_GPU
        amrex::Gpu::dtoh_memcpy(ids, d_ptr, size*sizeof(amrex::Long));
#endif
    }

    void artemis_convert_cpu_to_int (int* cpus, const ArtemisParticleContainer::ParticleType* pstructs, int size)
    {
        int* d_ptr = nullptr;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<int> d_cpus(size);
        d_ptr = d_cpus.data();
#else
        d_ptr = cpus;
#endif
        amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            d_ptr[i] = pstructs[i].cpu();
        });
#ifdef AMREX_USE_GPU
        amrex::Gpu::dtoh_memcpy(cpus, d_ptr, size*sizeof(int));
#endif
    }

    int artemis_getParticleCompIndex (
         const char* char_species_name, const char* char_comp_name )
    {
        const auto & mypc = Artemis::GetInstance().GetPartContainer();

        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);

        const std::string comp_name(char_comp_name);
        auto particle_comps = myspc.getParticleComps();

        return particle_comps.at(comp_name);
    }

    void artemis_addRealComp(const char* char_species_name,
        const char* char_comp_name, bool comm=true)
    {
        auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);

        const std::string comp_name(char_comp_name);
        myspc.AddRealComp(comp_name, comm);

        mypc.defineAllParticleTiles();
    }

    amrex::Real artemis_sumParticleCharge(const char* char_species_name, const bool local)
    {
        auto & mypc = Artemis::GetInstance().GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);
        return myspc.sumParticleCharge(local);
    }

    int artemis_getParticleBoundaryBufferSize(const char* species_name, int boundary, bool local)
    {
        const std::string name(species_name);
        auto& particle_buffers = Artemis::GetInstance().GetParticleBoundaryBuffer();
        return particle_buffers.getNumParticlesInContainer(species_name, boundary, local);
    }

    int** artemis_getParticleBoundaryBufferScrapedSteps(const char* species_name, int boundary, int lev,
                     int* num_tiles, int** particles_per_tile)
    {
        const std::string name(species_name);
        auto& particle_buffers = Artemis::GetInstance().GetParticleBoundaryBuffer();
        auto& particle_buffer = particle_buffers.getParticleBuffer(species_name, boundary);

        const int comp = particle_buffer.NumIntComps() - 1;

        *num_tiles = particle_buffer.numLocalTilesAtLevel(lev);
        *particles_per_tile = static_cast<int*>(malloc(*num_tiles*sizeof(int)));
        memset(*particles_per_tile, 0, *num_tiles*sizeof(int));

        auto data = static_cast<int**>(malloc(*num_tiles*sizeof(int*)));
        int i = 0;
        for (amrex::ParIter<0,0,PIdx::nattribs, 0, amrex::PinnedArenaAllocator> pti(particle_buffer, lev); pti.isValid(); ++pti, ++i) {
            auto& soa = pti.GetStructOfArrays();
            data[i] = (int*) soa.GetIntData(comp).dataPtr();
            (*particles_per_tile)[i] = pti.numParticles();
        }

        return data;
    }

    amrex::ParticleReal** artemis_getParticleBoundaryBuffer(const char* species_name, int boundary, int lev,
                     int* num_tiles, int** particles_per_tile, const char* comp_name)
    {
        const std::string name(species_name);
        auto& particle_buffers = Artemis::GetInstance().GetParticleBoundaryBuffer();
        auto& particle_buffer = particle_buffers.getParticleBuffer(species_name, boundary);

        const int comp = artemis_getParticleCompIndex(species_name, comp_name);

        *num_tiles = particle_buffer.numLocalTilesAtLevel(lev);
        *particles_per_tile = static_cast<int*>(malloc(*num_tiles*sizeof(int)));
        memset(*particles_per_tile, 0, *num_tiles*sizeof(int));

        auto data = static_cast<amrex::ParticleReal**>(malloc(*num_tiles*sizeof(amrex::ParticleReal*)));
        int i = 0;
        for (amrex::ParIter<0,0,PIdx::nattribs, 0, amrex::PinnedArenaAllocator> pti(particle_buffer, lev); pti.isValid(); ++pti, ++i) {
            auto& soa = pti.GetStructOfArrays();
            data[i] = (amrex::ParticleReal*) soa.GetRealData(comp).dataPtr();
            (*particles_per_tile)[i] = pti.numParticles();
        }

        return data;
    }

    amrex::ParticleReal** artemis_getParticleBoundaryBufferStructs(const char* species_name, int boundary, int lev,
                     int* num_tiles, int** particles_per_tile)
    {
        const std::string name(species_name);
        auto& particle_buffers = Artemis::GetInstance().GetParticleBoundaryBuffer();
        auto& particle_buffer = particle_buffers.getParticleBuffer(species_name, boundary);

        *num_tiles = particle_buffer.numLocalTilesAtLevel(lev);
        *particles_per_tile = static_cast<int*>(malloc(*num_tiles*sizeof(int)));
        memset(*particles_per_tile, 0, *num_tiles*sizeof(int));

        auto data = static_cast<amrex::ParticleReal**>(malloc(*num_tiles*sizeof(typename ArtemisParticleContainer::ParticleType*)));
        int i = 0;
        for (amrex::ParIter<0,0,PIdx::nattribs, 0, amrex::PinnedArenaAllocator> pti(particle_buffer, lev); pti.isValid(); ++pti, ++i) {
            auto& aos = pti.GetArrayOfStructs();
            data[i] = (amrex::ParticleReal*) aos.data();
            (*particles_per_tile)[i] = pti.numParticles();
        }
        return data;
    }

    void artemis_clearParticleBoundaryBuffer () {
        auto& particle_buffers = Artemis::GetInstance().GetParticleBoundaryBuffer();
        particle_buffers.clearParticles();
    }

    void artemis_depositChargeDensity (const char* char_species_name, int lev) {
        // this function is used to deposit a given species' charge density
        // in the rho_fp multifab which can then be accessed from python via
        // pyartemis.fields.RhoFPWrapper()
        Artemis& artemis = Artemis::GetInstance();
        const auto & mypc = artemis.GetPartContainer();
        const std::string species_name(char_species_name);
        auto & myspc = mypc.GetParticleContainerFromName(species_name);
        auto * rho_fp = artemis.get_pointer_rho_fp(lev);

        if (rho_fp == nullptr) {
            ablastr::warn_manager::WMRecordWarning(
                "ArtemisWrappers", "rho_fp is not allocated",
                ablastr::warn_manager::WarnPriority::low
            );
            return;
        }

        for (ArtemisParIter pti(myspc, lev); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            auto& wp = pti.GetAttribs(PIdx::w);
            // Do this unconditionally, ignoring myspc.do_not_deposit, to support diagnostic uses
            myspc.DepositCharge(pti, wp, nullptr, rho_fp, 0, 0, np, 0, lev, lev);
        }
#ifdef ARTEMIS_DIM_RZ
        artemis.ApplyInverseVolumeScalingToChargeDensity(rho_fp, lev);
#endif
    }

    void artemis_ComputeDt () {
        Artemis& artemis = Artemis::GetInstance();
        artemis.ComputeDt();
    }
    void artemis_MoveWindow (int step,bool move_j) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.MoveWindow(step, move_j);
    }

    void artemis_EvolveE (amrex::Real dt) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.EvolveE(dt);
    }
    void artemis_EvolveB (amrex::Real dt, DtType a_dt_type) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.EvolveB(dt, a_dt_type);
    }
    void artemis_FillBoundaryE () {
        Artemis& artemis = Artemis::GetInstance();
        artemis.FillBoundaryE(artemis.getngEB());
    }
    void artemis_FillBoundaryB () {
        Artemis& artemis = Artemis::GetInstance();
        artemis.FillBoundaryB(artemis.getngEB());
    }
    void artemis_SyncRho () {
        Artemis& artemis = Artemis::GetInstance();
        artemis.SyncRho();
    }
    void artemis_SyncCurrent (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.SyncCurrent(J_fp, J_cp);
    }
    void artemis_UpdateAuxilaryData () {
        Artemis& artemis = Artemis::GetInstance();
        artemis.UpdateAuxilaryData();
    }
    void artemis_PushParticlesandDepose (amrex::Real cur_time) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.PushParticlesandDepose(cur_time);
    }

    int artemis_getistep (int lev) {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.getistep(lev);
    }
    void artemis_setistep (int lev, int ii) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.setistep(lev, ii);
    }
    amrex::Real artemis_gett_new (int lev) {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.gett_new(lev);
    }
    void artemis_sett_new (int lev, amrex::Real time) {
        Artemis& artemis = Artemis::GetInstance();
        artemis.sett_new(lev, time);
    }
    amrex::Real artemis_getdt (int lev) {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.getdt(lev);
    }

    int artemis_maxStep () {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.maxStep();
    }
    amrex::Real artemis_stopTime () {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.stopTime();
    }

    int artemis_finestLevel () {
        Artemis& artemis = Artemis::GetInstance();
        return artemis.finestLevel();
    }

    int artemis_getMyProc () {
        return amrex::ParallelDescriptor::MyProc();
    }

    int artemis_getNProcs () {
        return amrex::ParallelDescriptor::NProcs();
    }

    void artemis_setPotentialEB (const char * char_potential) {
        Artemis& artemis = Artemis::GetInstance();
        const std::string potential(char_potential);
        artemis.m_poisson_boundary_handler.setPotentialEB(potential);
    }

    void mypc_Redistribute () {
        auto & mypc = Artemis::GetInstance().GetPartContainer();
        mypc.Redistribute();
    }
