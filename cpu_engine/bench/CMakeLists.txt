add_executable(affinity_bench affinity_bench.cpp)
target_include_directories(affinity_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h
)
target_compile_features(affinity_bench PRIVATE cxx_std_23)
set_project_warnings(affinity_bench)

if(UNIX AND NOT APPLE)
    target_link_libraries(affinity_bench PRIVATE pthread)
endif()
