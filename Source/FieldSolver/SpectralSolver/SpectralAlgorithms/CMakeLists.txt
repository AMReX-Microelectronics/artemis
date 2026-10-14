target_sources(Artemis
  PRIVATE
    PsatdAlgorithmFirstOrder.cpp
    PsatdAlgorithmJConstantInTime.cpp
    PsatdAlgorithmJLinearInTime.cpp
    PsatdAlgorithmPml.cpp
    SpectralBaseAlgorithm.cpp
    PsatdAlgorithmComoving.cpp
)

if(Artemis_DIMS STREQUAL RZ)
    target_sources(Artemis
      PRIVATE
        SpectralBaseAlgorithmRZ.cpp
        PsatdAlgorithmRZ.cpp
        PsatdAlgorithmGalileanRZ.cpp
        PsatdAlgorithmPmlRZ.cpp
    )
endif()
