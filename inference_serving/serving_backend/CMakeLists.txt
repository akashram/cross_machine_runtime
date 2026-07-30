# Step 8: backend-agnostic serving router — portable, builds and runs
# everywhere including this Mac. Only the CPU backend can actually
# execute here; GPU/FPGA/TPU are registered as real, honestly-
# unavailable entries so the router's fallback logic has real backends
# to fall back from.

add_library(ServingRouter serving_router.cpp)
target_include_directories(ServingRouter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ServingRouter PUBLIC Transformer)
target_compile_features(ServingRouter PUBLIC cxx_std_23)
set_project_warnings(ServingRouter)

add_executable(serving_router_test serving_router_test.cpp)
target_link_libraries(serving_router_test PRIVATE ServingRouter)
set_project_warnings(serving_router_test)

# Real long-running process closing the gap Phase 16's K8s manifests
# disclosed: k8s/serving/deployment.yaml needs an actual binary to run,
# not just the ServingRouter library the unit test links against.
add_executable(serving_daemon serving_daemon.cpp)
target_link_libraries(serving_daemon PRIVATE ServingRouter)
set_project_warnings(serving_daemon)

enable_testing()
add_test(NAME serving_router_test COMMAND serving_router_test)
