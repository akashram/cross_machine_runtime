# Step 16: Topology-aware scheduler — portable placement algorithm over
# an abstract topology graph. Builds and runs everywhere including this
# Mac; real PCIe/NVLink/EFA discovery (populating a TopologyGraph) is a
# separate, Linux/GPU-hardware-specific concern not implemented here.

add_library(TopoScheduler topo_scheduler.cpp)
target_include_directories(TopoScheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(topo_scheduler_test topo_scheduler_test.cpp)
target_link_libraries(topo_scheduler_test PRIVATE TopoScheduler)

enable_testing()
add_test(NAME topo_scheduler_test COMMAND topo_scheduler_test)
