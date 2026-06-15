# cmake/Cuda.cmake
# Standard nvcc compile options for all gpu_engine targets.
# Usage: include(Cuda) then call target_apply_cuda_flags(mytarget)

function(target_apply_cuda_flags target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -lineinfo
            --ptxas-options=-v
            -Xcompiler=-Wall,-Wextra
        >
    )
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
            -G -g -O0
        >
    )
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
            -O3 --use_fast_math
        >
    )
endfunction()
