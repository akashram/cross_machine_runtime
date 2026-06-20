#pragma once
#include <cstddef>
// TODO: implement on multi-GPU hardware

struct P2PBandwidthResult {
    int src_device;
    int dst_device;
    bool nvlink;          // true if NVLink, false if PCIe
    double bandwidth_gbs; // GB/s
    double latency_us;
};

// Enable peer access between src and dst (bidirectional).
// Returns false if peer access is not supported.
bool enable_peer_access(int src, int dst);

// Measure P2P bandwidth for a transfer of `bytes` from src to dst.
P2PBandwidthResult measure_p2p_bandwidth(int src, int dst, size_t bytes);

// Measure host-staged bandwidth (GPU→pinned host→GPU).
P2PBandwidthResult measure_host_staged_bandwidth(int src, int dst, size_t bytes);
