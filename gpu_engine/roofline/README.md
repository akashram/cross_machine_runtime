# GPU Roofline Model

**Status: STUB — requires CUDA GPU. Run on target instance before benchmarking kernels.**

## What this measures
Measures peak FLOPS (via cuBLAS) and peak HBM bandwidth (via memory bandwidth test),
then plots achieved FLOPS vs. the roofline ceiling for every kernel. Classifies each
kernel as compute-bound or bandwidth-bound.

## Implementation notes
- Peak FLOPS: `cublasSgemmStridedBatched` with maximum occupancy config
- Peak HBM bandwidth: large memcpy kernel (STREAM TRIAD), compare to vendor spec
- Arithmetic intensity (AI) = FLOPS / bytes_transferred
- Roofline ceiling = min(peak_FLOPS, AI * peak_bandwidth)
- Kernel is compute-bound if achieved FLOPS ≈ peak_FLOPS
- Kernel is bandwidth-bound if achieved FLOPS ≈ AI * peak_bandwidth

## Results

TODO: run on GPU hardware and fill in this table.

### Hardware ceilings

| Metric | Measured | Vendor spec | % of peak |
|--------|----------|-------------|-----------|
| Peak FP32 TFLOPS | TODO | TODO | TODO |
| Peak FP16 TFLOPS | TODO | TODO | TODO |
| Peak HBM bandwidth (GB/s) | TODO | TODO | TODO |

### Kernel roofline analysis

| Kernel | AI (FLOP/byte) | Achieved TFLOPS | Bound by | % of ceiling |
|--------|----------------|-----------------|----------|--------------|
| elementwise_add | TODO | TODO | bandwidth | TODO |
| gemm_naive | TODO | TODO | TODO | TODO |
| gemm_tiled | TODO | TODO | TODO | TODO |
| gemm_wmma | TODO | TODO | compute | TODO |
| flash_attn_fwd | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU
- Build preset: cuda release (Linux)
- Run this step first on a new instance to calibrate before benchmarking kernels
