set(RUNTIME_PGO "" CACHE STRING
  "PGO mode: 'instrument' (step 1) or 'use' (step 3)")

set(PGO_PROFILE_PATH "${CMAKE_SOURCE_DIR}/build/pgo.profdata" CACHE PATH
  "Path to merged .profdata file used when RUNTIME_PGO=use")

if(RUNTIME_PGO STREQUAL "instrument")
  message(STATUS "PGO: instrumentation build (-fprofile-instr-generate)")
  add_compile_options(-fprofile-instr-generate)
  add_link_options(-fprofile-instr-generate)
elseif(RUNTIME_PGO STREQUAL "use")
  if(NOT EXISTS "${PGO_PROFILE_PATH}")
    message(FATAL_ERROR
      "PGO profile not found at ${PGO_PROFILE_PATH}\n"
      "Run the instrument build and collect profiles first (see cpu_engine/pgo/run_pgo.sh).")
  endif()
  message(STATUS "PGO: optimised build using ${PGO_PROFILE_PATH}")
  add_compile_options(-fprofile-instr-use=${PGO_PROFILE_PATH})
  add_link_options(-fprofile-instr-use=${PGO_PROFILE_PATH})
endif()
