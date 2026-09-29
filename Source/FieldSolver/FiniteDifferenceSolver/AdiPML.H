/* Convolutional (CFS) PML auxiliaries for the macroscopic ADI Maxwell solver.
 *
 * Unsplit CPML on the ADI grid (not split-field WarpX pml[lev] boxes).
 * With warpx.do_pml_in_domain = 0 (default), GrowDomainForAdiOutsidePML()
 * extends geometry.prob_* / amr.n_cell by pml_ncell outside the physical
 * box before grids are created — same user-facing model as explicit FDTD
 * outside-domain PML, but still one contiguous BoxArray for ADI pencils.
 * With do_pml_in_domain = 1, the outer pml_ncell of the user domain absorb.
 */
#ifndef WARPX_ADI_PML_H_
#define WARPX_ADI_PML_H_

namespace AdiPsiE
{
    enum { EXY = 0, EXZ, EYX, EYZ, EZX, EZY };
}

namespace AdiPsiH
{
    enum { HXY = 0, HXZ, HYX, HYZ, HZX, HZY };
}

namespace amrex { class MultiFab; }

// quantity: 0 = kappa, 1 = a, 2 = b, 3 = 1/kappa.
// Samples at dst's actual Yee location along `dir` (cell center or node).
// The field update uses ψ from the start of the half-step and 1/κ.
// Auxiliaries advance after E and B, using a and b.
void FillAdiCfsOnLayout (amrex::MultiFab& dst, int dir, int quantity);

#endif
