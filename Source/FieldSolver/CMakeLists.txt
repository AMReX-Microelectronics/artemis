target_sources(Artemis
  PRIVATE
    ElectrostaticSolver.cpp
    ArtemisPushFieldsEM.cpp
    Artemis_QED_Field_Pushers.cpp
    ArtemisExternalEMFields.cpp
)

add_subdirectory(FiniteDifferenceSolver)
add_subdirectory(London)
add_subdirectory(LumpedElement)
add_subdirectory(MagnetostaticSolver)
if(Artemis_PSATD)
    add_subdirectory(SpectralSolver)
endif()
