/* Copyright 2022 Axel Huebl
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Artemis.H"
#include "Utils/ArtemisVersion.H"

#include <string>


std::string
Artemis::Version ()
{
    std::string version;
#ifdef ARTEMIS_GIT_VERSION
    version = std::string(ARTEMIS_GIT_VERSION);
#endif
    if( version.empty() )
        return std::string("Unknown");
    else
        return version;
}

std::string
Artemis::PicsarVersion ()
{
    std::string version;
#ifdef PICSAR_GIT_VERSION
    version = std::string(PICSAR_GIT_VERSION);
#endif
    if( version.empty() )
        return std::string("Unknown");
    else
        return version;
}
