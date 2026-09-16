# Phase 21 step 8: storage bandwidth/IOPS QoS, built on the real
# networking/multitenancy Multitenancy library target (unmodified).

add_executable(storage_qos_test storage_qos_test.cpp)
target_include_directories(storage_qos_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(storage_qos_test PRIVATE Multitenancy)
set_project_warnings(storage_qos_test)

enable_testing()
add_test(NAME storage_qos_test COMMAND storage_qos_test)
