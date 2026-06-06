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

add_executable(avx512_bench avx512_bench.cpp)
target_include_directories(avx512_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(avx512_bench PRIVATE cxx_std_23)
set_project_warnings(avx512_bench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(avx512_bench PRIVATE -mavx2)
    if(RUNTIME_AVX512)
        target_compile_options(avx512_bench PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(avx512_bench PRIVATE pthread)
endif()

add_executable(branchless_bench branchless_bench.cpp)
target_include_directories(branchless_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(branchless_bench PRIVATE cxx_std_23)
set_project_warnings(branchless_bench)
if(UNIX AND NOT APPLE)
    target_link_libraries(branchless_bench PRIVATE pthread)
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

add_executable(tiling_bench tiling_bench.cpp)
target_include_directories(tiling_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # tiling/matmul.h, affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h, foundation/perf/perf.h
)
target_compile_features(tiling_bench PRIVATE cxx_std_23)
set_project_warnings(tiling_bench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(tiling_bench PRIVATE -mavx2)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(tiling_bench PRIVATE pthread)
endif()

add_executable(mlp_bench mlp_bench.cpp)
target_include_directories(mlp_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # inference/mlp.h, affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h, foundation/perf/perf.h
)
target_compile_features(mlp_bench PRIVATE cxx_std_23)
set_project_warnings(mlp_bench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(mlp_bench PRIVATE -mavx2)
    if(RUNTIME_AVX512)
        target_compile_options(mlp_bench PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(mlp_bench PRIVATE pthread)
endif()

add_executable(roofline_bench roofline_bench.cpp)
target_include_directories(roofline_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # roofline/, avx512/, tiling/, inference/, affinity/
    ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h
)
target_compile_features(roofline_bench PRIVATE cxx_std_23)
set_project_warnings(roofline_bench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(roofline_bench PRIVATE -mavx2 -mfma)
    if(RUNTIME_AVX512)
        target_compile_options(roofline_bench PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(roofline_bench PRIVATE pthread)
endif()
