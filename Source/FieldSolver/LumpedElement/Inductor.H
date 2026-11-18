/*
 * This file is part of MicroEleX
 *
 * License: BSD-3-Clause-LBNL
 */


#ifndef INDUCTOR_H
#define INDUCTOR_H

#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>


class Inductor {

public:
    //Constructor
    Inductor ();

    void ReadParameters ();
    void InitData ();
    void EvolveInductorJ (amrex::Real dt);

    void InitializeInductorMultiFabUsingParser ( amrex::MultiFab *inductor_mf,
                                                 amrex::ParserExecutor<3> const& inductor_parser, const int lev);

    std::string m_str_inductor_x_function;
    std::string m_str_inductor_y_function;
    std::string m_str_inductor_z_function;
    std::unique_ptr<amrex::Parser> m_inductor_x_parser;
    std::unique_ptr<amrex::Parser> m_inductor_y_parser;
    std::unique_ptr<amrex::Parser> m_inductor_z_parser;
    std::unique_ptr<amrex::MultiFab> m_inductor_x_mf;
    std::unique_ptr<amrex::MultiFab> m_inductor_y_mf;
    std::unique_ptr<amrex::MultiFab> m_inductor_z_mf;

     /** Gpu Vector with index type of the jx multifab */
     amrex::GpuArray<int, 3> jx_IndexType;
     /** Gpu Vector with index type of the jy multifab */
     amrex::GpuArray<int, 3> jy_IndexType;
     /** Gpu Vector with index type of the jz multifab */
     amrex::GpuArray<int, 3> jz_IndexType;

};


#endif
