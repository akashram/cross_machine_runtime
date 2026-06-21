# gpu_engine/ptx_sass — Step 10: PTX/SASS inspection workflow
#
# No build targets: cuobjdump and ncu operate on existing binaries from
# other steps (elementwise_bench, gemm_bench, etc.).
#
# inspect.sh: dumps PTX + SASS from any CUDA binary, runs sass_analyze.py.
# sass_analyze.py: counts known inefficiency patterns (spills, scalar loads,
#   NOPs, barriers) and prints a summary table per kernel.
#
# To get the best analysis, rebuild the binary of interest with:
#   -Xptxas -v                 (register/smem per kernel at link time)
#   --generate-line-info       (source ↔ PTX ↔ SASS correlation in ncu)
# These can be added to a kernel's target_compile_options in its CMakeLists.

# Make scripts executable at configure time
file(CHMOD "${CMAKE_CURRENT_SOURCE_DIR}/inspect.sh"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                 GROUP_READ GROUP_EXECUTE
                 WORLD_READ WORLD_EXECUTE)

file(CHMOD "${CMAKE_CURRENT_SOURCE_DIR}/sass_analyze.py"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                 GROUP_READ GROUP_EXECUTE
                 WORLD_READ WORLD_EXECUTE)
