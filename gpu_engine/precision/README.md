# Mixed Precision + Tensor Core Alignment

**Status: STUB — requires A100/H100 for BF16/FP8. Run on p4d.24xlarge (A100).**

## What this measures
- BF16 forward pass + FP32 master weights + dynamic loss scaling
- FP8 forward pass on Hopper (H100)
- Tensor Core alignment analysis: demonstrate perf cliff when M/N/K not multiples of 8/16/32

## Implementation notes

### Mixed precision
- BF16: supported on Ampere+; `__bfloat16` type in CUDA
- FP32 master weights: optimizer holds FP32 copy, casts to BF16 for forward/backward
- Loss scaling: scale loss by 2^k before backward to avoid FP16/BF16 underflow;
  halve scale on overflow (NaN/Inf in gradients), double scale on N consecutive clean steps
- Validate: compare loss curves BF16 vs FP32 on toy model (should converge identically)

### Tensor Core alignment
- FP16/BF16 WMMA: M/N/K must be multiples of 16
- TF32 (Ampere): multiples of 8 (uses FP32 precision)
- FP8 (Hopper): multiples of 16
- Cliff measurement: sweep M from 1 to 256 in steps of 1, plot TFLOPS vs M

## Results

TODO: run on GPU hardware and fill in this table.

### Mixed precision forward pass (MLP, batch=256)

| Dtype | Latency (ms) | Memory (GB) | Loss matches FP32? |
|-------|-------------|-------------|---------------------|
| FP32 | TODO | TODO | reference |
| BF16 | TODO | TODO | TODO |
| FP8 (Hopper only) | TODO | TODO | TODO |

### Tensor Core alignment cliff

| M (N=K=4096) | TFLOPS | Notes |
|-------------|--------|-------|
| 16 | TODO | aligned |
| 17 | TODO | unaligned — cliff? |
| 32 | TODO | aligned |
| 64 | TODO | aligned |
| 128 | TODO | aligned |

## Hardware notes
- BF16: Ampere+ (A100, H100); A10G, T4 do NOT support BF16 Tensor Cores
- FP8: Hopper only (H100) — p5.48xlarge
- Build preset: cuda (Linux)
