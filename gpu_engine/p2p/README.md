# GPUDirect P2P

**Status: STUB — requires multi-GPU instance. Run on p3.8xlarge (4x V100) or p4d.24xlarge (8x A100).**

## What this measures
Enables CUDA peer access between GPUs and benchmarks direct GPU-to-GPU transfer
via `cudaMemcpyPeerAsync`, comparing against host-staged transfer (GPU→CPU→GPU).

## Implementation notes
- Enable: `cudaDeviceEnablePeerAccess(dst, 0)` (both directions)
- Check: `cudaDeviceCanAccessPeer(&can, src, dst)` — depends on PCIe/NVLink topology
- P2P bandwidth via NVLink (A100): ~600 GB/s bidirectional
- P2P bandwidth via PCIe: ~16–32 GB/s (PCIe 4.0 x16)
- Host-staged: GPU→pinned host memory→GPU, limited to PCIe bandwidth (~32 GB/s)
- Key question: on NVLink topology, P2P should be ~10–20x faster than host-staged

## Results

TODO: run on GPU hardware and fill in this table.

| Transfer type | Size | Bandwidth (GB/s) | Latency (µs) |
|---------------|------|-----------------|--------------|
| Host-staged (GPU0→CPU→GPU1) | 1 GB | TODO | TODO |
| P2P (GPU0→GPU1, PCIe) | 1 GB | TODO | TODO |
| P2P (GPU0→GPU1, NVLink) | 1 GB | TODO | TODO |

## Hardware notes
- Required: multi-GPU instance (p3.8xlarge+)
- NVLink: only on p3/p4 instances; g4dn has no NVLink
- Build preset: cuda (Linux)
