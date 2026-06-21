#pragma once
// GPUDirect P2P — direct GPU-to-GPU copies without staging through host memory.
//
// On NVLink-connected GPUs (p3.8xlarge, p4d, DGX systems), cudaMemcpyPeerAsync
// transfers data directly through the NVLink fabric at ~600 GB/s aggregate
// (A100 NVLink-3) vs ~32 GB/s PCIe 4.0 × 16.
//
// On PCIe-only systems (most g4dn/p2 instances), peer access still works but
// routes through the PCIe bus, matching PCIe bandwidth rather than bypassing it.
//
// This header measures both and produces a comparison table so you can verify
// whether NVLink is active and quantify the benefit over host-staged transfer.
//
// NVLink detection heuristic: if peer bandwidth > 100 GB/s, assume NVLink.
// For precise topology, use nvidia-smi topo --matrix.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

struct P2PBandwidthResult {
    int    src_device;
    int    dst_device;
    bool   nvlink;           // heuristic: bandwidth > 100 GB/s → NVLink
    double bandwidth_gbs;    // GB/s unidirectional
    double latency_us;       // µs per transfer
};

namespace detail {

#define P2P_CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
} while (0)

inline float time_copies(int dst, void* d_dst, int src, const void* d_src,
                          size_t bytes, int iters) {
    // Events on the dst device for timing
    cudaSetDevice(dst);
    cudaEvent_t t0, t1;
    P2P_CUDA_CHECK(cudaEventCreate(&t0));
    P2P_CUDA_CHECK(cudaEventCreate(&t1));
    // Warm up
    P2P_CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes));
    P2P_CUDA_CHECK(cudaDeviceSynchronize());

    P2P_CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        P2P_CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes));
    P2P_CUDA_CHECK(cudaEventRecord(t1));
    P2P_CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    P2P_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    P2P_CUDA_CHECK(cudaEventDestroy(t0));
    P2P_CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

} // namespace detail

// Enable bidirectional peer access between src and dst.
// Returns false if the hardware does not support peer access for this pair.
inline bool enable_peer_access(int src, int dst) {
    int can = 0;
    detail::P2P_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, src, dst));
    if (!can) return false;

    int dev_count = 0;
    detail::P2P_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (src >= dev_count || dst >= dev_count) return false;

    cudaSetDevice(src);
    cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) return false;

    cudaSetDevice(dst);
    e = cudaDeviceEnablePeerAccess(src, 0);
    return (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
}

// Measure direct P2P bandwidth for a transfer of `bytes` from src to dst.
// Requires enable_peer_access(src, dst) to have been called first.
inline P2PBandwidthResult measure_p2p_bandwidth(int src, int dst, size_t bytes) {
    void *d_src, *d_dst;
    cudaSetDevice(src);
    detail::P2P_CUDA_CHECK(cudaMalloc(&d_src, bytes));
    detail::P2P_CUDA_CHECK(cudaMemset(d_src, 0xab, bytes));
    cudaSetDevice(dst);
    detail::P2P_CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    constexpr int ITERS = 20;
    float ms = detail::time_copies(dst, d_dst, src, d_src, bytes, ITERS);

    P2PBandwidthResult r;
    r.src_device    = src;
    r.dst_device    = dst;
    r.bandwidth_gbs = bytes / (ms / 1e3) / 1e9;
    r.latency_us    = ms * 1e3;  // ms → µs
    r.nvlink        = r.bandwidth_gbs > 100.0;  // PCIe4 ×16 peaks at ~32 GB/s

    cudaSetDevice(src); detail::P2P_CUDA_CHECK(cudaFree(d_src));
    cudaSetDevice(dst); detail::P2P_CUDA_CHECK(cudaFree(d_dst));
    return r;
}

// Measure host-staged bandwidth: src→pinned host memory→dst.
// Baseline to compare against direct P2P.
inline P2PBandwidthResult measure_host_staged_bandwidth(int src, int dst, size_t bytes) {
    void *h_pinned;
    detail::P2P_CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

    void *d_src, *d_dst;
    cudaSetDevice(src);
    detail::P2P_CUDA_CHECK(cudaMalloc(&d_src, bytes));
    detail::P2P_CUDA_CHECK(cudaMemset(d_src, 0xcd, bytes));
    cudaSetDevice(dst);
    detail::P2P_CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // Warm up
    detail::P2P_CUDA_CHECK(cudaMemcpy(h_pinned, d_src, bytes, cudaMemcpyDeviceToHost));
    detail::P2P_CUDA_CHECK(cudaMemcpy(d_dst, h_pinned, bytes, cudaMemcpyHostToDevice));
    detail::P2P_CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int ITERS = 20;
    using Clock = std::chrono::high_resolution_clock;
    // Use chrono since staged transfers involve two cudaMemcpy calls and there
    // is no single event to bracket them.
    auto t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        cudaSetDevice(src);
        detail::P2P_CUDA_CHECK(cudaMemcpy(h_pinned, d_src, bytes, cudaMemcpyDeviceToHost));
        cudaSetDevice(dst);
        detail::P2P_CUDA_CHECK(cudaMemcpy(d_dst, h_pinned, bytes, cudaMemcpyHostToDevice));
    }
    auto t1 = Clock::now();
    detail::P2P_CUDA_CHECK(cudaDeviceSynchronize());

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    P2PBandwidthResult r;
    r.src_device    = src;
    r.dst_device    = dst;
    r.nvlink        = false;
    r.bandwidth_gbs = bytes / (ms / 1e3) / 1e9;
    r.latency_us    = ms * 1e3;

    detail::P2P_CUDA_CHECK(cudaFreeHost(h_pinned));
    cudaSetDevice(src); detail::P2P_CUDA_CHECK(cudaFree(d_src));
    cudaSetDevice(dst); detail::P2P_CUDA_CHECK(cudaFree(d_dst));
    return r;
}
