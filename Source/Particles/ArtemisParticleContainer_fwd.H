/* Copyright 2021 Luca Fedeli, Axel Huebl
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ARTEMIS_ArtemisParticleContainer_fwd_H_
#define ARTEMIS_ArtemisParticleContainer_fwd_H_

#include <AMReX_GpuAllocators.H>


enum struct ParticleBC;
enum struct ConvertDirection;

struct PIdx;
struct DiagIdx;
struct TmpIdx;

class ArtemisParIter;

class ArtemisParticleContainer;

struct DiagIdx
{
    enum {
        w = 0,
        x, y, z, ux, uy, uz,
        nattribs
    };
};

struct TmpIdx
{
    enum {
        xold = 0,
        yold, zold, uxold, uyold, uzold,
        nattribs
    };
};

#endif /* ARTEMIS_ArtemisParticleContainer_fwd_H_ */
