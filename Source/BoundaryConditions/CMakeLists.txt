target_sources(Artemis
  PRIVATE
    PML.cpp
    ArtemisEvolvePML.cpp
    ArtemisFieldBoundaries.cpp
    Artemis_PEC.cpp
)

if(Artemis_DIMS STREQUAL RZ)
    target_sources(Artemis
      PRIVATE
        PML_RZ.cpp
    )
endif()
