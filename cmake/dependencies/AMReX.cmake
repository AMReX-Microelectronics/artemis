macro(find_amrex)
    if(WarpX_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        include(FetchContent)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
        if(WarpX_ASCENT)
            set(AMReX_ASCENT ON CACHE INTERNAL "")
            set(AMReX_CONDUIT ON CACHE INTERNAL "")
        endif()

        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(AMReX_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(AMReX_FPE ON CACHE BOOL "")
        else()
            set(AMReX_ASSERTIONS OFF CACHE BOOL "")
            set(AMReX_FPE OFF CACHE BOOL "")
        endif()

        if(WarpX_COMPUTE STREQUAL CUDA)
            set(AMReX_CUDA  ON  CACHE INTERNAL "")
            set(AMReX_DPCPP OFF CACHE BOOL "")
            set(AMReX_HIP   OFF CACHE BOOL "")
            set(AMReX_OMP   OFF CACHE INTERNAL "")
        elseif(WarpX_COMPUTE STREQUAL OMP)
            set(AMReX_CUDA  OFF CACHE INTERNAL "")
            set(AMReX_DPCPP OFF CACHE BOOL "")
            set(AMReX_HIP   OFF CACHE BOOL "")
            set(AMReX_OMP   ON  CACHE INTERNAL "")
        elseif(WarpX_COMPUTE STREQUAL DPCPP)
            set(AMReX_CUDA  OFF CACHE INTERNAL "")
            set(AMReX_DPCPP ON  CACHE BOOL "")
            set(AMReX_HIP   OFF CACHE BOOL "")
            set(AMReX_OMP   OFF CACHE INTERNAL "")
        elseif(WarpX_COMPUTE STREQUAL HIP)
            set(AMReX_CUDA  OFF CACHE INTERNAL "")
            set(AMReX_DPCPP OFF  CACHE BOOL "")
            set(AMReX_HIP   ON CACHE BOOL "")
            set(AMReX_OMP   OFF CACHE INTERNAL "")
        else()
            set(AMReX_CUDA  OFF CACHE INTERNAL "")
            set(AMReX_DPCPP OFF CACHE BOOL "")
            set(AMReX_HIP   OFF CACHE BOOL "")
            set(AMReX_OMP   OFF CACHE INTERNAL "")
        endif()

        if(WarpX_MPI)
            set(AMReX_MPI ON CACHE INTERNAL "")
            if(WarpX_MPI_THREAD_MULTIPLE)
                set(AMReX_MPI_THREAD_MULTIPLE ON CACHE INTERNAL "")
            else()
                set(AMReX_MPI_THREAD_MULTIPLE OFF CACHE INTERNAL "")
            endif()
        else()
            set(AMReX_MPI OFF CACHE INTERNAL "")
        endif()

        if(WarpX_PRECISION STREQUAL "DOUBLE")
            set(AMReX_PRECISION "DOUBLE" CACHE INTERNAL "")
            set(AMReX_PRECISION_PARTICLES "DOUBLE" CACHE INTERNAL "")
        else()
            set(AMReX_PRECISION "SINGLE" CACHE INTERNAL "")
            set(AMReX_PRECISION_PARTICLES "SINGLE" CACHE INTERNAL "")
        endif()

        set(AMReX_FORTRAN OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
        set(AMReX_PARTICLES ON CACHE INTERNAL "")
        set(AMReX_TINY_PROFILE ON CACHE BOOL "")

        # AMReX_SENSEI
        # we'll need this for Python bindings
        #set(AMReX_PIC ON CACHE INTERNAL "")

        if(WarpX_DIMS STREQUAL RZ)
            set(AMReX_SPACEDIM 2 CACHE INTERNAL "")
        else()
            set(AMReX_SPACEDIM ${WarpX_DIMS} CACHE INTERNAL "")
        endif()

        FetchContent_Declare(fetchedamrex
            GIT_REPOSITORY ${WarpX_amrex_repo}
            GIT_TAG        ${WarpX_amrex_branch}
            BUILD_IN_SOURCE 0
        )
        FetchContent_GetProperties(fetchedamrex)

        if(NOT fetchedamrex_POPULATED)
            FetchContent_Populate(fetchedamrex)
            list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")
            if(AMReX_CUDA)
                enable_language(CUDA)
                include(AMReX_SetupCUDA)
            endif()
            add_subdirectory(${fetchedamrex_SOURCE_DIR} ${fetchedamrex_BINARY_DIR})
        endif()

        # advanced fetch options
        mark_as_advanced(FETCHCONTENT_BASE_DIR)
        mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_QUIET)
        mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)

        # AMReX options not relevant to WarpX users
        mark_as_advanced(AMREX_BUILD_DATETIME)
        mark_as_advanced(AMReX_SPACEDIM)
        mark_as_advanced(AMReX_ASSERTIONS)
        mark_as_advanced(AMReX_AMRDATA)
        mark_as_advanced(AMReX_BASE_PROFILE) # mutually exclusive to tiny profile
        mark_as_advanced(AMReX_CONDUIT)
        mark_as_advanced(AMReX_CUDA)
        mark_as_advanced(AMReX_PARTICLES)
        mark_as_advanced(AMReX_PARTICLES_PRECISION)
        mark_as_advanced(AMReX_DPCPP)
        mark_as_advanced(AMReX_EB)
        mark_as_advanced(AMReX_FPE)
        mark_as_advanced(AMReX_FORTRAN)
        mark_as_advanced(AMReX_FORTRAN_INTERFACES)
        mark_as_advanced(AMReX_HDF5)  # we do HDF5 I/O (and more) via openPMD-api
        mark_as_advanced(AMReX_HIP)
        mark_as_advanced(AMReX_LINEAR_SOLVERS)
        mark_as_advanced(AMReX_MEM_PROFILE)
        mark_as_advanced(AMReX_MPI)
        mark_as_advanced(AMReX_MPI_THREAD_MULTIPLE)
        mark_as_advanced(AMReX_OMP)
        mark_as_advanced(AMReX_PIC)
        mark_as_advanced(AMReX_SENSEI)
        mark_as_advanced(AMReX_TINY_PROFILE)
        mark_as_advanced(AMReX_TP_PROFILE)
        mark_as_advanced(USE_XSDK_DEFAULTS)

        message(STATUS "AMReX: Using INTERNAL version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
    else()
        # https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#importing-amrex-into-your-cmake-project
        if(WarpX_ASCENT)
            set(COMPONENT_ASCENT AMReX_ASCENT AMReX_CONDUIT)
        else()
            set(COMPONENT_ASCENT)
        endif()
        if(WarpX_DIMS STREQUAL RZ)
            set(COMPONENT_DIM 2D)
        else()
            set(COMPONENT_DIM ${WarpX_DIMS}D)
        endif()
        set(COMPONENT_PRECISION ${WarpX_PRECISION} P${WarpX_PRECISION})

        find_package(AMReX 20.11 CONFIG REQUIRED COMPONENTS ${COMPONENT_ASCENT} ${COMPONENT_DIM} PARTICLES ${COMPONENT_PRECISION} TINYP LSOLVERS)
        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
    endif()
endmacro()

set(WarpX_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(WarpX_amrex_internal)")
set(WarpX_amrex_branch "development"
    CACHE STRING
    "Repository branch for WarpX_amrex_repo if(WarpX_amrex_internal)")

find_amrex()