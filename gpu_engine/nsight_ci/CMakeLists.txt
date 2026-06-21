# gpu_engine/nsight_ci — Step 24: Nsight Compute CI integration
#
# No compiled binary — this step provides scripts and tooling:
#
#   ncu_profile.sh: runs ncu --set full on a benchmark binary, writes CSV + JSON + .ncu-rep
#   parse_ncu.py:   parses ncu CSV into structured JSON with derived metrics
#                   (TFLOPS, HBM BW, AI, coalescing efficiency, occupancy)
#                   and optional regression detection vs a baseline JSON.
#
# Usage in CI (after building the target):
#   bash gpu_engine/nsight_ci/ncu_profile.sh ./build/release/roofline_bench
#   bash gpu_engine/nsight_ci/ncu_profile.sh ./build/release/gemm_bench gemm \
#       --baseline gpu_engine/nsight_ci/baselines/gemm.json
#
# Install scripts to build output so they're alongside the benchmarks
install(PROGRAMS ncu_profile.sh parse_ncu.py
        DESTINATION bin/nsight_ci
        OPTIONAL)

# Create baselines directory (for storing per-commit reference JSONs)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/nsight)
