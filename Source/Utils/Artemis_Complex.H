/* Copyright 2019-2020 Andrew Myers, David Grote, Maxence Thevenet
 * Remi Lehe
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_COMPLEX_H_
#define ARTEMIS_COMPLEX_H_

#ifdef ARTEMIS_USE_PSATD
#  include "FieldSolver/SpectralSolver/AnyFFT.H"
#endif

#include <AMReX_Gpu.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_REAL.H>

#include <complex>

// Defines a complex type on GPU & CPU
using Complex = amrex::GpuComplex<amrex::Real>;

#ifdef ARTEMIS_USE_PSATD
static_assert(sizeof(Complex) == sizeof(AnyFFT::Complex),
    "The complex type in Artemis and the FFT library do not match.");
#endif

static_assert(sizeof(Complex) == sizeof(amrex::Real[2]),
              "Unexpected complex type.");

#endif //ARTEMIS_COMPLEX_H_
