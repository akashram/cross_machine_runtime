# Step 6: automated fault injection + recovery suite — portable, builds
# and runs everywhere including this Mac. Reuses networking/raft's real
# RaftNode and fpga_engine/thermal_router's real (hardware-free)
# ThermalRouter decision logic; no new component invented here.

add_executable(chaos_suite chaos_suite.cpp ${CMAKE_SOURCE_DIR}/fpga_engine/thermal_router/thermal_policy.cpp)
target_include_directories(chaos_suite PRIVATE ${CMAKE_SOURCE_DIR}/fpga_engine/thermal_router)
target_link_libraries(chaos_suite PRIVATE Raft)
set_project_warnings(chaos_suite)

enable_testing()
add_test(NAME chaos_suite COMMAND chaos_suite)
