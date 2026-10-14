/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ARTEMIS_PROFILERWRAPPER_H_
#define ARTEMIS_PROFILERWRAPPER_H_

#include "Artemis.H"
#include "ablastr/profiler/ProfilerWrapper.H"


// `BL_PROFILE_PASTE(SYNC_SCOPE_, __COUNTER__)` and `SYNC_V_##vname` used to make unique names for
// synchronizeOnDestruct objects, like `SYNC_SCOPE_0` and `SYNC_V_pmain`
#define ARTEMIS_PROFILE(fname) ABLASTR_PROFILE(fname, Artemis::do_device_synchronize)
#define ARTEMIS_PROFILE_VAR(fname, vname) ABLASTR_PROFILE_VAR(fname, vname, Artemis::do_device_synchronize)
#define ARTEMIS_PROFILE_VAR_NS(fname, vname) ABLASTR_PROFILE_VAR_NS(fname, vname, Artemis::do_device_synchronize)
#define ARTEMIS_PROFILE_VAR_START(vname) ABLASTR_PROFILE_VAR_START(vname, Artemis::do_device_synchronize)
#define ARTEMIS_PROFILE_VAR_STOP(vname) ABLASTR_PROFILE_VAR_STOP(vname, Artemis::do_device_synchronize)
#define ARTEMIS_PROFILE_REGION(rname) ABLASTR_PROFILE_REGION(rname, Artemis::do_device_synchronize)

#endif // ARTEMIS_PROFILERWRAPPER_H_
