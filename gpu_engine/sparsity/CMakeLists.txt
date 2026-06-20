# Step 20: 2:4 structured sparsity — stub, requires A100/H100 + cuSPARSELt
# TODO: uncomment when on GPU hardware
# find_library(CUSPARSELT_LIB cusparselt)
# add_cuda_benchmark(sparsity_bench sparsity_bench.cu)
# target_link_libraries(sparsity_bench PRIVATE CUDA::cusparse ${CUSPARSELT_LIB})
