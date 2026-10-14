/* Copyright 2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ARTEMIS_DIAGNOSTICS_REDUCEDDIAGS_FIELDMAXIMUM_H_
#define ARTEMIS_DIAGNOSTICS_REDUCEDDIAGS_FIELDMAXIMUM_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class mainly contains a function that computes the maximum value of each component
 * of the EM field and of the vector norm of the E and B fields.
 */
class FieldMaximum : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    FieldMaximum(std::string rd_name);

    /**
     * This function computes the maximum value of Ex, Ey, Ez, |E|, Bx, By, Bz and |B|
     *
     * @param[in] step current time step
     */
    virtual void ComputeDiags(int step) override final;

};

#endif // ARTEMIS_DIAGNOSTICS_REDUCEDDIAGS_FIELDMAXIMUM_H_
