foreach(target pgo_train pgo_measure)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..    # avx512/, tiling/, inference/, affinity/
        ${CMAKE_SOURCE_DIR}               # foundation/bench/bench.h
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    set_project_warnings(${target})
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(${target} PRIVATE -mavx2)
        if(RUNTIME_AVX512)
            target_compile_options(${target} PRIVATE
                -mavx512f -mavx512bw -mavx512vl -mavx512dq)
        endif()
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PRIVATE pthread)
    endif()
endforeach()
