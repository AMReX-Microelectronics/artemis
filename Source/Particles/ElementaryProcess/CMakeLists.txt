target_sources(Artemis
  PRIVATE
    Ionization.cpp
)

if(Artemis_QED)
    target_sources(Artemis
      PRIVATE
        QEDPairGeneration.cpp
        QEDPhotonEmission.cpp
    )
    add_subdirectory(QEDInternals)
endif()
