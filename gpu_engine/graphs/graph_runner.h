#pragma once
// CUDA Graph runner — capture a workload once, replay at near-zero CPU cost.
//
// Problem: every eager kernel launch has a ~5–20 µs CPU overhead (driver
// round-trip, argument marshalling, dependency checking).  For models with
// hundreds of small kernels, this overhead accumulates and the CPU becomes
// the bottleneck even when the GPU is underloaded.
//
// CUDA Graphs fix this by:
//   1. Capturing all kernel launches into a directed acyclic graph during a
//      dedicated capture pass on a stream.
//   2. Instantiating the graph into an executable (cudaGraphExec_t).
//   3. Replaying the graph with a single `cudaGraphLaunch` call — one CPU
//      submission that schedules the entire subgraph.
//
// Caveat: the graph captures the exact kernel arguments and dependencies
// recorded during capture.  If pointers or shapes change between iterations,
// the graph must be re-captured (or updated via cudaGraphExecKernelNodeSetParams
// for lightweight changes).
//
// API:
//   GraphRunner runner;
//   runner.capture(stream, [&]{ my_forward_pass(); });  // capture once
//   for each training step:
//       runner.replay(stream);   // near-zero CPU overhead

#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <cuda_runtime.h>

#define GR_CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
} while (0)

class GraphRunner {
public:
    GraphRunner() = default;

    // Capture `workload` as a CUDA graph on `stream`.
    // workload() must only enqueue operations on `stream` (or child streams
    // created within the capture region).  CPU-side operations (malloc, printf,
    // conditional branches on GPU results) are not captured.
    void capture(cudaStream_t stream, std::function<void()> workload) {
        destroy_existing();

        // Begin capture: all operations on `stream` from this point are
        // recorded rather than submitted to the GPU.
        GR_CUDA_CHECK(cudaStreamBeginCapture(stream,
                                             cudaStreamCaptureModeGlobal));
        workload();
        GR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));

        // Instantiate: validate the graph and compile it into an executable.
        GR_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_,
                                           nullptr, nullptr, 0));
    }

    // Replay the captured graph.  Returns immediately on the CPU; the GPU
    // executes asynchronously.  Sync with cudaStreamSynchronize if needed.
    void replay(cudaStream_t stream) {
        if (!graph_exec_)
            throw std::runtime_error("GraphRunner::replay: no graph captured");
        GR_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    }

    // Measure average CPU overhead per submission.
    // Compares:
    //   eager   — `n_iterations` individual kernel launches via `eager_fn`
    //   graph   — `n_iterations` graph replays via replay(stream)
    // Both paths do the same GPU work; only CPU scheduling cost differs.
    struct OverheadResult {
        double eager_launch_us;   // CPU µs per eager call (total / n_iters)
        double graph_replay_us;   // CPU µs per graph replay
        double speedup;           // eager / graph
    };

    OverheadResult measure_overhead(cudaStream_t stream,
                                    std::function<void()> eager_fn,
                                    int n_iterations = 1000) {
        if (!graph_exec_)
            throw std::runtime_error("measure_overhead: no graph captured");

        // Warm up (prime kernel cache, avoid cold-start bias)
        for (int i = 0; i < 10; ++i) eager_fn();
        GR_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Time eager submissions (CPU time only — do NOT sync to GPU inside)
        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();
        for (int i = 0; i < n_iterations; ++i) eager_fn();
        auto t1 = Clock::now();
        GR_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Warm up graph
        for (int i = 0; i < 10; ++i) replay(stream);
        GR_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Time graph replays
        auto t2 = Clock::now();
        for (int i = 0; i < n_iterations; ++i) replay(stream);
        auto t3 = Clock::now();
        GR_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto us = [](auto a, auto b) {
            return std::chrono::duration<double, std::micro>(b - a).count();
        };

        OverheadResult r;
        r.eager_launch_us = us(t0, t1) / n_iterations;
        r.graph_replay_us = us(t2, t3) / n_iterations;
        r.speedup         = r.eager_launch_us / r.graph_replay_us;
        return r;
    }

    // Print a human-readable summary of the captured graph.
    void print_info() const {
        if (!graph_) { printf("GraphRunner: no graph captured\n"); return; }
        size_t num_nodes = 0;
        cudaGraphGetNodes(graph_, nullptr, &num_nodes);
        printf("GraphRunner: %zu node(s) in captured graph\n", num_nodes);
    }

    ~GraphRunner() { destroy_existing(); }

    // Not copyable (owning CUDA handles)
    GraphRunner(const GraphRunner&)            = delete;
    GraphRunner& operator=(const GraphRunner&) = delete;

private:
    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    void destroy_existing() {
        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_)      { cudaGraphDestroy(graph_);           graph_      = nullptr; }
    }
};
