add_executable(bench_main bench_main.cpp)

target_include_directories(bench_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(bench_main PRIVATE cxx_std_23)

set_project_warnings(bench_main)
