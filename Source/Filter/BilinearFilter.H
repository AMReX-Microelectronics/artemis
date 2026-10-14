/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_BILIN_FILTER_H_
#define ARTEMIS_BILIN_FILTER_H_

#include "Filter.H"

#include <AMReX_Array.H>
#include <AMReX_Config.H>

class BilinearFilter : public Filter
{
public:

    BilinearFilter() = default;

    void ComputeStencils();

    amrex::Array<unsigned int, AMREX_SPACEDIM> npass_each_dir;

};
#endif // #ifndef ARTEMIS_BILIN_FILTER_H_
