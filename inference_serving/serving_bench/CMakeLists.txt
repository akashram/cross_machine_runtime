# Step 9: serving benchmarks — this repo's own CPU-path latency/throughput
# is real and run locally (serving_latency_bench, links against Transformer
# + ServingRouter, no GPU dependency). The vLLM/TensorRT-LLM comparison
# (vllm_tensorrt_bench.py) is hardware-gated Python, no CMake target.

add_executable(serving_latency_bench serving_latency_bench.cpp)
target_link_libraries(serving_latency_bench PRIVATE ServingRouter Transformer)
set_project_warnings(serving_latency_bench)
