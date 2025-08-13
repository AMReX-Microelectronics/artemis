/* Copyright 2024 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */


#ifndef FERROE_H
#define FERROE_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>


class FerroE {

public:
    //Constructor
    FerroE ();

    void ReadParameters ();
    void InitData ();
    void EvolveFerroEJ (amrex::Real dt);
    void EvolveP (amrex::Real dt);

    void InitializeFerroelectricMultiFabUsingParser ( amrex::MultiFab *sc_mf,
                                                       amrex::ParserExecutor<3> const& sc_parser, const int lev);

    std::string m_str_ferroelectric_function;
    std::unique_ptr<amrex::Parser> m_ferroelectric_parser;
    std::unique_ptr<amrex::MultiFab> m_ferroelectric_mf;

    int include_Landau = 1;
    int include_grad = 0;


};

#endif
