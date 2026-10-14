/* Copyright 2020 Axel Huebl
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ARTEMIS_MPI_INIT_HELPERS_H_
#define ARTEMIS_MPI_INIT_HELPERS_H_

#include <utility>

namespace utils
{
    /** Return the required MPI threading
     *
     * @return the MPI_THREAD_* level required for MPI_Init_thread
     */
    int
    artemis_mpi_thread_required ();

    /** Initialize MPI
     *
     * @return pair(required, provided) of MPI thread level from MPI_Init_thread
     */
    std::pair< int, int >
    artemis_mpi_init (int argc, char* argv[]);

    /** Finalize MPI
     *
     * This function is simply a wrapper around MPI_Finalize(). It is a no-op if
     * Artemis is compiled without MPI support.
     */
    void
    artemis_mpi_finalize ();

    /** Check if the requested MPI thread level is valid
     *
     * Prints warnings and notes otherwise.
     */
    void
    artemis_check_mpi_thread_level ();

} // namespace utils

#endif // ARTEMIS_MPI_INIT_HELPERS_H_
