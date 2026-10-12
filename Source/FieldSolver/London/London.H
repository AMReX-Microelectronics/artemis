/* Copyright 2020 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */


#ifndef LONDON_H
#define LONDON_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>


class London {

public:
    //Constructor
    London ();

    void ReadParameters ();
    void InitData ();
    void EvolveLondonJ (amrex::Real dt);

    void InitializeSuperconductorMultiFabUsingParser ( amrex::MultiFab *sc_mf,
                                                       amrex::ParserExecutor<3> const& sc_parser, const int lev);

    amrex::Real m_penetration_depth;
    std::string m_str_superconductor_function;
    std::unique_ptr<amrex::Parser> m_superconductor_parser;
    std::unique_ptr<amrex::MultiFab> m_superconductor_mf;

     /** Gpu Vector with index type of the jx multifab */
     amrex::GpuArray<int, 3> jx_IndexType;
     /** Gpu Vector with index type of the jy multifab */
     amrex::GpuArray<int, 3> jy_IndexType;
     /** Gpu Vector with index type of the jz multifab */
     amrex::GpuArray<int, 3> jz_IndexType;

};


#endif
