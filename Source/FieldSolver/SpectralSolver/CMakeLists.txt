target_sources(Artemis
  PRIVATE
    SpectralFieldData.cpp
    SpectralKSpace.cpp
    SpectralSolver.cpp
)

if(Artemis_COMPUTE STREQUAL CUDA)
    target_sources(ablastr PRIVATE WrapCuFFT.cpp)
elseif(Artemis_COMPUTE STREQUAL HIP)
    target_sources(ablastr PRIVATE WrapRocFFT.cpp)
else()
    target_sources(ablastr PRIVATE WrapFFTW.cpp)
endif()

if(Artemis_DIMS STREQUAL RZ)
    target_sources(Artemis
      PRIVATE
        SpectralSolverRZ.cpp
        SpectralFieldDataRZ.cpp
        SpectralKSpaceRZ.cpp
        SpectralBinomialFilter.cpp
    )
    add_subdirectory(SpectralHankelTransform)
endif()

add_subdirectory(SpectralAlgorithms)
