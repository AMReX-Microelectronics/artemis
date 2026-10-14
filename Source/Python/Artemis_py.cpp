/* Copyright 2019 David Grote, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Artemis_py.H"

std::map< std::string, ARTEMIS_CALLBACK_PY_FUNC_0 > artemis_callback_py_map;

bool IsPythonCallBackInstalled ( std::string name )
{
    return (artemis_callback_py_map.count(name) == 1u);
}

// Execute Python callbacks of the type given by the input string
void ExecutePythonCallback ( std::string name )
{
    if ( IsPythonCallBackInstalled(name) ) {
        ARTEMIS_PROFILE("artemis_py_"+name);
        artemis_callback_py_map[name]();
    }
}
