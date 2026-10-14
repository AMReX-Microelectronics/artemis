target_sources(Artemis
  PRIVATE
    Interpolate.cpp
    MPIInitHelpers.cpp
    ParticleUtils.cpp
    RelativeCellPosition.cpp
    ArtemisAlgorithmSelection.cpp
    ArtemisMovingWindow.cpp
    ArtemisTagging.cpp
    ArtemisUtil.cpp
    ArtemisrocfftUtil.cpp
    ArtemisVersion.cpp
    cnpy.cpp
)

add_subdirectory(Logo)
add_subdirectory(Parser)
