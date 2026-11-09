#include "WarpX.H"
#include "BoundaryConditions/PML.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Evolve/WarpXDtType.H"
#include "WarpX_PEC.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <memory>
using namespace amrex::literals;
using namespace amrex;

void WarpX::ApplyEfieldBoundary(const int lev, PatchType patch_type)
{
    if (PEC::isAnyBoundaryPEC()) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoEfield( { get_pointer_Efield_fp(lev, 0),
                                     get_pointer_Efield_fp(lev, 1),
                                     get_pointer_Efield_fp(lev, 2) }, lev, patch_type);
            if (WarpX::isAnyBoundaryPML()) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoEfield( pml[lev]->GetE_fp(), lev, patch_type, split_pml_field);
            }
        } else {
            PEC::ApplyPECtoEfield( { get_pointer_Efield_cp(lev, 0),
                                     get_pointer_Efield_cp(lev, 1),
                                     get_pointer_Efield_cp(lev, 2) }, lev, patch_type);
            if (WarpX::isAnyBoundaryPML()) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoEfield( pml[lev]->GetE_cp(), lev, patch_type, split_pml_field);
            }
        }
    }
}

void WarpX::ApplyPECMaskToEfield (const int lev, const bool include_guard_cells)
{
    // The PEC mask is only initialized by the macroscopic solver (MacroscopicProperties)
    if (!use_PEC_mask || em_solver_medium != MediumForEM::Macroscopic) { return; }

    // Grid: the guard cells of PEC_fp are filled at initialization
    // (see MacroscopicProperties::InitData), so the mask can also be applied
    // to the E guard cells, e.g., after an excitation has written into them.
    for (int idim = 0; idim < 3; ++idim) {
        const amrex::IntVect ng = include_guard_cells ?
            amrex::min(Efield_fp[lev][idim]->nGrowVect(), PEC_fp[lev][idim]->nGrowVect()) :
            amrex::IntVect::TheZeroVector();
        amrex::MultiFab::Multiply(*Efield_fp[lev][idim], *PEC_fp[lev][idim], 0, 0, 1, ng);
    }

    // PML: mask extended into the PML (algo.extend_PEC_mask_into_pml)
    if (do_pml && extend_PEC_mask_into_pml && pml[lev] && pml[lev]->ok()) {
        const auto E_pml   = pml[lev]->GetE_fp();
        const auto PEC_pml = pml[lev]->GetPEC_fp();

        // Split-field PML: both split components are zeroed on the PEC edges.
        // CPML: component 0 holds the total field (component 1 is zero), so the
        // same multiply enforces E_tan = 0 on the PEC mask extended into the PML.
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Multiply(*E_pml[idim], *PEC_pml[idim], 0, 0, 2, 0);
        }

        if (pml[lev]->isConvolutional()) {
            // Also reset the CPML memory variables on PEC edges, so that no
            // stale convolution history is carried by masked E components
            const auto psiE = pml[lev]->GetPsiE_fp();
            for (int idim = 0; idim < 3; ++idim) {
                amrex::MultiFab::Multiply(*psiE[idim], *PEC_pml[idim], 0, 0, 1, 0);
                amrex::MultiFab::Multiply(*psiE[idim], *PEC_pml[idim], 0, 1, 1, 0);
            }
        }
    }
}

void WarpX::ApplyBfieldBoundary (const int lev, PatchType patch_type, DtType a_dt_type)
{
    if (PEC::isAnyBoundaryPEC()) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoBfield( { get_pointer_Bfield_fp(lev, 0),
                                     get_pointer_Bfield_fp(lev, 1),
                                     get_pointer_Bfield_fp(lev, 2) }, lev, patch_type);
        } else {
            PEC::ApplyPECtoBfield( { get_pointer_Bfield_cp(lev, 0),
                                     get_pointer_Bfield_cp(lev, 1),
                                     get_pointer_Bfield_cp(lev, 2) }, lev, patch_type);
        }
    }

    // Silver-Mueller boundaries are only applied on the first half-push of B
    // This is because the formula used for Silver-Mueller assumes that
    // E and B are staggered in time, which is only true after the first half-push
    if (lev == 0) {
        if (a_dt_type == DtType::FirstHalf) {
            bool applySilverMueller = false;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if ( (WarpX::field_boundary_lo[idim] == FieldBoundaryType::Absorbing_SilverMueller) ||
                   (WarpX::field_boundary_hi[idim] == FieldBoundaryType::Absorbing_SilverMueller) ) {
                    applySilverMueller = true;
                }
            }
            if(applySilverMueller) m_fdtd_solver_fp[0]->ApplySilverMuellerBoundary(
                                         Efield_fp[lev], Bfield_fp[lev],
                                         Geom(lev).Domain(), dt[lev],
                                         WarpX::field_boundary_lo,
                                         WarpX::field_boundary_hi);
        }
    }
}

void WarpX::ApplyRhofieldBoundary (const int lev, MultiFab* rho,
                                   PatchType patch_type)
{
    if (PEC::isAnyBoundaryPEC()) PEC::ApplyPECtoRhofield(rho, lev, patch_type);
}

void WarpX::ApplyJfieldBoundary (const int lev, amrex::MultiFab* Jx,
                                 amrex::MultiFab* Jy, amrex::MultiFab* Jz,
                                 PatchType patch_type)
{
    if (PEC::isAnyBoundaryPEC()) PEC::ApplyPECtoJfield(Jx, Jy, Jz, lev, patch_type);
}
