/* Copyright 2019 David Grote, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_PY_H_
#define ARTEMIS_PY_H_

#include "ArtemisWrappers.H"
#include "Utils/ArtemisProfilerWrapper.H"

#include <map>
#include <string>

/**
 * Declare global map to hold python callback functions.
 *
 * The keys of the map describe at what point in the simulation the python
 * functions will be called. Currently supported keys (callback points) are
 * afterinit, beforecollisions, aftercollisions, beforeEsolve, poissonsolver,
 * afterEsolve, beforeInitEsolve, afterInitEsolve,
 * beforedeposition, afterdeposition, particlescraper,
 * particleloader, beforestep, afterstep, afterdiagnostics, afterrestart,
 * oncheckpointsignal, onbreaksignal, particleinjection and appliedfields.
*/
extern std::map< std::string, ARTEMIS_CALLBACK_PY_FUNC_0 > artemis_callback_py_map;

/**
 * \brief Function to check if the given name is a key in artemis_callback_py_map
 */
bool IsPythonCallBackInstalled ( std::string name );

/**
 * \brief Function to look for and execute Python callbacks
 */
void ExecutePythonCallback ( std::string name );

#endif
