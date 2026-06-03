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

add_executable(prefetch_bench prefetch_bench.cpp)
target_include_directories(prefetch_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(prefetch_bench PRIVATE cxx_std_23)
set_project_warnings(prefetch_bench)
if(UNIX AND NOT APPLE)
    target_link_libraries(prefetch_bench PRIVATE pthread)
endif()

add_executable(nt_store_bench nt_store_bench.cpp)
target_include_directories(nt_store_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # nt_store/, affinity/, hugepage/
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h, foundation/perf/perf.h
)
target_compile_features(nt_store_bench PRIVATE cxx_std_23)
set_project_warnings(nt_store_bench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(nt_store_bench PRIVATE -mavx2)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(nt_store_bench PRIVATE pthread)
endif()
