add_executable(latency_bench latency_bench.cpp)
target_include_directories(latency_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # affinity/affinity.h
    ${CMAKE_SOURCE_DIR}               # foundation/spsc_queue.h, foundation/bench/bench.h
)
target_compile_features(latency_bench PRIVATE cxx_std_23)
set_project_warnings(latency_bench)

# pthread required for std::thread on Linux; implicit on macOS
if(UNIX AND NOT APPLE)
    target_link_libraries(latency_bench PRIVATE pthread)
endif()
