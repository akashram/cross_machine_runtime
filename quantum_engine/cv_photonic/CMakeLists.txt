# Phase 20 step 8: continuous-variable/photonic primitives. Header-only,
# zero dependencies (no state_vector dependency either -- Gaussian states
# are a genuinely separate formalism).

add_library(QuantumCVPhotonic INTERFACE)
target_include_directories(QuantumCVPhotonic INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cv_photonic_test cv_photonic_test.cpp)
target_link_libraries(cv_photonic_test PRIVATE QuantumCVPhotonic)
set_project_warnings(cv_photonic_test)

add_test(NAME cv_photonic_test COMMAND cv_photonic_test)
