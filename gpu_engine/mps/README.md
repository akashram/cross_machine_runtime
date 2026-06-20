# CUDA MPS (Multi-Process Service)

**Status: STUB — requires CUDA GPU + Linux. Run on g4dn.xlarge or better.**

## What this measures
Configures CUDA MPS server and measures context-switching overhead reduction
when multiple processes share a GPU. Without MPS, each process gets a separate
CUDA context (expensive switch). With MPS, all processes share one context.

## Implementation notes
- Setup: `nvidia-smi -i 0 -c EXCLUSIVE_PROCESS` then `nvidia-cuda-mps-control -d`
- Teardown: `echo quit | nvidia-cuda-mps-control`
- Measure: N processes each launching small kernels — time from first launch to last completion
- Expected: MPS reduces context-switching overhead from ~10µs to ~1µs per switch
- Also measure: MPS throughput when N processes run simultaneously vs time-sharing

## Results

TODO: run on GPU hardware and fill in this table.

| N processes | Without MPS (ms/kernel) | With MPS (ms/kernel) | Speedup |
|-------------|------------------------|----------------------|---------|
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU on Linux (MPS not supported on Windows)
- Build preset: cuda (Linux)
- Run setup_mps.sh before benchmarking
