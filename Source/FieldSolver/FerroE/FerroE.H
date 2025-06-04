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
    // amrex::Real compute_ex_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz);
    // amrex::Real compute_ey_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz);
    // amrex::Real compute_ez_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz);

    void InitializeFerroelectricMultiFabUsingParser ( amrex::MultiFab *sc_mf,
                                                       amrex::ParserExecutor<3> const& sc_parser, const int lev);

    std::string m_str_ferroelectric_function;
    std::unique_ptr<amrex::Parser> m_ferroelectric_parser;
    std::unique_ptr<amrex::MultiFab> m_ferroelectric_mf;

    int include_Landau = 1;
    int include_grad = 0;

    //Coefficients in the evolution of the polarization equation
    // const amrex::Real mu = 1.35e-18;
    // const amrex::Real gamma = 2.0e-7;
    //Gradient energy coefficients
    // const amrex::Real G_11 = 5.1e-10;
    
    //Landau free energy coefficients
    // const amrex::Real alpha_1 = -3.7116e7;
    // const amrex::Real alpha_11 = -2.097e8;
    // const amrex::Real alpha_12 = 7.974e8;
    // const amrex::Real alpha_111 = 1.294e9;
    // const amrex::Real alpha_112 = -1.95e9;
    // const amrex::Real alpha_123 = -2.5e9;
    // const amrex::Real alpha_1111 = 3.863e10;
    // const amrex::Real alpha_1112 = 2.529e10;
    // const amrex::Real alpha_1122 = 1.637e10;
    // const amrex::Real alpha_1123 = 1.367e10;


};

#endif
