# Occupancy Tuner

**Status: STUB — requires CUDA GPU. Run on g4dn.xlarge or better.**

## What this measures
Uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to sweep block sizes and
shared memory per block, measuring active warps/SM as a function of register and
shared memory usage. Documents the tradeoffs per kernel.

## Implementation notes
- Query: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockDim, sharedMem)`
- Theoretical occupancy = blocks * blockDim / max_threads_per_sm
- Sweep block sizes: 32, 64, 128, 256, 512, 1024
- For each block size, vary shared memory from 0 to 48KB
- Also use `cudaFuncGetAttributes` to read register count per thread
- Target: find block size that maximizes occupancy without register spilling

## Results

TODO: run on GPU hardware and fill in this table.

| Kernel | Block Size | Shared Mem (KB) | Registers/Thread | Occupancy (%) | Throughput (GB/s or TFLOPS) |
|--------|-----------|-----------------|------------------|---------------|------------------------------|
| add_kernel | TODO | TODO | TODO | TODO | TODO |
| gemm_tiled | TODO | TODO | TODO | TODO | TODO |
| flash_attn | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU
- Build preset: cuda (Linux)
- Note: Occupancy ≠ performance. Document cases where lower occupancy wins due to register reuse.
