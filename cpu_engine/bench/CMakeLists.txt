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

add_executable(hugepage_bench hugepage_bench.cpp)
target_include_directories(hugepage_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # hugepage/hugepage.h, affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h, foundation/perf/perf.h
)
target_compile_features(hugepage_bench PRIVATE cxx_std_23)
set_project_warnings(hugepage_bench)

if(UNIX AND NOT APPLE)
    target_link_libraries(hugepage_bench PRIVATE pthread)
endif()

add_executable(jitter_bench jitter_bench.cpp)
target_include_directories(jitter_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h
)
target_compile_features(jitter_bench PRIVATE cxx_std_23)
set_project_warnings(jitter_bench)

if(UNIX AND NOT APPLE)
    target_link_libraries(jitter_bench PRIVATE pthread)
endif()
