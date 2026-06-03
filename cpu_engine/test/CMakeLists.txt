add_executable(affinity_test affinity_test.cpp)
target_include_directories(affinity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(affinity_test PRIVATE cxx_std_23)
set_project_warnings(affinity_test)
add_test(NAME affinity_test COMMAND affinity_test)

if(UNIX AND NOT APPLE)
    target_link_libraries(affinity_test PRIVATE pthread)
endif()
