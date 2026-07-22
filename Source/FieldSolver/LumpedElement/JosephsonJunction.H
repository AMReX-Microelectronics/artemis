/*
 * This file is part of MicroEleX
 *
 * License: BSD-3-Clause-LBNL
 */


#ifndef JOSEPHSON_JUNCTION_H
#define JOSEPHSON_JUNCTION_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>


/** \brief Josephson junction (nonlinear inductor).
 *
 * Sets the supercurrent term  J_J = Ic * sin(phi) / A  on every edge where
 * Ic(x,y,z) != 0.  The phase phi obeys the second Josephson relation
 *     d phi / dt = (2e / hbar) * V,   V = E_edge * edge_length
 * advanced with leapfrog
 *     phi^{n+1/2} = phi^{n-1/2} + dt * (2e/hbar) * V^n.
 * Because the current MultiFab is a persistent state in this solver (it is not
 * re-zeroed each step), the junction updates J by the *difference* between the
 * new and previously-added supercurrent so that J holds the instantaneous
 * Ic*sin(phi)/A, not a running sum. This also composes with any London /
 * inductor current sharing the same edge.
 *
 * Linear resistive shunt (R) and junction capacitance (C) of an RCSJ model are
 * NOT handled in this class — set them via macroscopic.lumped_resistor_*
 * and macroscopic.lumped_capacitor_* on the same cell (see MacroscopicProperties).
 */
class JosephsonJunction {

public:
    JosephsonJunction ();

    void ReadParameters ();
    void InitData ();

    /** Advance phi by dt using E^n, then add Ic*sin(phi)/A to the J edges.
     *  Pass -0.5*dt for the leapfrog initialization push, dt for the main loop. */
    void EvolveJunctionJ (amrex::Real dt);

    void InitializeJunctionMultiFabUsingParser (amrex::MultiFab *mf,
                                                amrex::ParserExecutor<3> const& parser,
                                                int lev);

    /** Persist the junction state (phi, Js) across a restart.
     *  Without these, a restart silently resets phi to 0 while the EM fields are
     *  restored from the checkpoint, leaving the junction phase inconsistent with
     *  the fields (the JJ loses its history and the post-restart dynamics are wrong).
     *  Call WriteCheckpoint when the checkpoint is flushed, and ReadCheckpoint AFTER
     *  InitData() on restart (InitData allocates and zeroes these MultiFabs). */
    void WriteCheckpoint (const std::string& dir) const;
    void ReadCheckpoint (const std::string& dir);

    // Parser string (from input file)
    std::string m_str_Ic_x_function;
    std::string m_str_Ic_y_function;
    std::string m_str_Ic_z_function;

    std::unique_ptr<amrex::Parser> m_Ic_x_parser;
    std::unique_ptr<amrex::Parser> m_Ic_y_parser;
    std::unique_ptr<amrex::Parser> m_Ic_z_parser;

    /** Critical current Ic [A] per edge. Non-zero entries mark junction edges. */
    std::unique_ptr<amrex::MultiFab> m_Ic_x_mf;
    std::unique_ptr<amrex::MultiFab> m_Ic_y_mf;
    std::unique_ptr<amrex::MultiFab> m_Ic_z_mf;

    /** Persistent phase state phi [rad], same staggering as J. */
    std::unique_ptr<amrex::MultiFab> m_phi_x_mf;
    std::unique_ptr<amrex::MultiFab> m_phi_y_mf;
    std::unique_ptr<amrex::MultiFab> m_phi_z_mf;

    /** Supercurrent density Ic*sin(phi)/A [A/m^2] added to J on the previous
     *  push. J is a persistent state in this solver (never re-zeroed), so we
     *  subtract this on the next push and add the new value — J then carries the
     *  instantaneous supercurrent rather than a running sum over time steps. */
    std::unique_ptr<amrex::MultiFab> m_Js_x_mf;
    std::unique_ptr<amrex::MultiFab> m_Js_y_mf;
    std::unique_ptr<amrex::MultiFab> m_Js_z_mf;

    amrex::GpuArray<int, 3> jx_IndexType;
    amrex::GpuArray<int, 3> jy_IndexType;
    amrex::GpuArray<int, 3> jz_IndexType;

};


#endif
