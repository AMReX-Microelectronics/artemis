/* Copyright 2023 Luca Fedeli
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ARTEMIS_ROCFFT_UTIL_H_
#define ARTEMIS_ROCFFT_UTIL_H_

namespace utils::rocfft
{
    /** This function is a wrapper around rocff_setup().
     *  It is a no-op in case rocfft is not used.
    */
    void setup();

    /** This function is a wrapper around rocff_cleanup().
     *  It is a no-op in case rocfft is not used.
    */
    void cleanup();
}

#endif //ARTEMIS_ROCFFT_UTILS_H_
