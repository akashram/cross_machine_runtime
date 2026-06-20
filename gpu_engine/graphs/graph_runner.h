#pragma once
#include <functional>
#include <cuda_runtime.h>
// TODO: implement on GPU hardware

class GraphRunner {
public:
    // Capture a workload into a CUDA graph on the given stream.
    void capture(cudaStream_t stream, std::function<void()> workload);

    // Replay the captured graph. Near-zero CPU overhead per call.
    void replay(cudaStream_t stream);

    // Measure average CPU overhead for N replays vs N eager launches (µs).
    struct OverheadResult {
        double eager_launch_us;  // CPU time to submit N kernels eagerly
        double graph_replay_us;  // CPU time to replay graph N times
        double speedup;
    };
    OverheadResult measure_overhead(int n_iterations);

    ~GraphRunner();

private:
    cudaGraph_t     graph_     = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};
