/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef PARTICLEBOUNDARIES_K_H_
#define PARTICLEBOUNDARIES_K_H_

#include "ParticleBoundaries.H"

#include <AMReX_AmrCore.H>

namespace ApplyParticleBoundaries {

    /* \brief Applies the boundary condition on a specific axis
     *        This is called by apply_boundaries.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundary (amrex::ParticleReal& x, amrex::Real xmin, amrex::Real xmax,
                    bool& change_sign_ux, bool& particle_lost,
                    ParticleBoundaryType xmin_bc, ParticleBoundaryType xmax_bc,
                    amrex::Real refl_probability_xmin, amrex::Real refl_probability_xmax,
                    amrex::RandomEngine const& engine )
    {
        if (x < xmin) {
            if (xmin_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmin_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmin == 0 || amrex::Random(engine) > refl_probability_xmin) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmin - x;
                    change_sign_ux = true;
                }
            }
            else if (xmin_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmin - x;
                change_sign_ux = true;
            }
        }
        else if (x > xmax) {
            if (xmax_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmax_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmax == 0 || amrex::Random(engine) > refl_probability_xmax) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmax - x;
                    change_sign_ux = true;
                }
            }
            else if (xmax_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmax - x;
                change_sign_ux = true;
            }
        }
    }

    /* \brief Applies absorbing or reflecting boundary condition to the input particles, along all axis.
     *        For reflecting boundaries, the position of the particle is changed appropriately and
     *        the sign of the velocity is changed (depending on the reflect_all_velocities flag).
     *        For absorbing, a flag is set whether the particle has been lost (it is up to the calling
     *        code to take appropriate action to remove any lost particles). Absorbing boundaries can
     *        be given a reflection coefficient for stochastic reflection of particles, this
     *        coefficient is zero by default.
     *        Note that periodic boundaries are handled in AMReX code.
     *
     * \param x, xmin, xmax: particle x position, location of x boundary
     * \param y, ymin, ymax: particle y position, location of y boundary (3D only)
     * \param z, zmin, zmax: particle z position, location of z boundary
     * \param ux, uy, uz: particle momenta
     * \param particle_lost: output, flags whether the particle was lost
     * \param boundaries: object with boundary condition settings
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundaries (
#ifndef WARPX_DIM_1D_Z
              amrex::ParticleReal& x, amrex::Real xmin, amrex::Real xmax,
#endif
#if (defined WARPX_DIM_3D) || (defined WARPX_DIM_RZ)
                      amrex::ParticleReal& y,
#endif
#if (defined WARPX_DIM_3D)
                      amrex::Real ymin, amrex::Real ymax,
#endif
                      amrex::ParticleReal& z, amrex::Real zmin, amrex::Real zmax,
                      amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
                      bool& particle_lost,
                      ParticleBoundaries::ParticleBoundariesData const& boundaries,
                      amrex::RandomEngine const& engine)
    {
        bool change_sign_ux = false;
        bool change_sign_uy = false;
        bool change_sign_uz = false;

#ifndef WARPX_DIM_1D_Z
        apply_boundary(x, xmin, xmax, change_sign_ux, particle_lost,
                       boundaries.xmin_bc, boundaries.xmax_bc,
                       boundaries.reflection_model_xlo(-ux), boundaries.reflection_model_xhi(ux),
                       engine);
#endif
#ifdef WARPX_DIM_3D
        apply_boundary(y, ymin, ymax, change_sign_uy, particle_lost,
                       boundaries.ymin_bc, boundaries.ymax_bc,
                       boundaries.reflection_model_ylo(-uy), boundaries.reflection_model_yhi(uy),
                       engine);
#endif
        apply_boundary(z, zmin, zmax, change_sign_uz, particle_lost,
                       boundaries.zmin_bc, boundaries.zmax_bc,
                       boundaries.reflection_model_zlo(-uz), boundaries.reflection_model_zhi(uz),
                       engine);

        if (boundaries.reflect_all_velocities && (change_sign_ux | change_sign_uy | change_sign_uz)) {
            change_sign_ux = true;
            change_sign_uy = true;
            change_sign_uz = true;
        }
#ifdef WARPX_DIM_RZ
        // Note that the reflection of the position does "r = 2*rmax - r", but this is only approximate.
        // The exact calculation requires the position at the start of the step.
        if (change_sign_ux && change_sign_uy) {
            ux = -ux;
            uy = -uy;
        } else if (change_sign_ux) {
            // Reflect only ur
            // Note that y is theta
            amrex::Real ur = ux*std::cos(y) + uy*std::sin(y);
            amrex::Real ut = -ux*std::sin(y) + uy*std::cos(y);
            ur = -ur;
            ux = ur*std::cos(y) - ut*std::sin(y);
            uy = ur*std::sin(y) + ut*std::cos(y);
        }
#else
        if (change_sign_ux) ux = -ux;
        if (change_sign_uy) uy = -uy;
#endif
        if (change_sign_uz) uz = -uz;
    }

}
#endif
