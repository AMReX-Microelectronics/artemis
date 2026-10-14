target_sources(Artemis
  PRIVATE
    FlushFormatAscent.cpp
    FlushFormatCheckpoint.cpp
    FlushFormatPlotfile.cpp
    FlushFormatSensei.cpp
)

if(Artemis_HAVE_OPENPMD)
    target_sources(Artemis
      PRIVATE
        FlushFormatOpenPMD.cpp
    )
endif()
