# CUDA Graphs

**Status: STUB — requires CUDA GPU. Run on g4dn.xlarge or better.**

## What this measures
Captures a full forward pass as a CUDA graph and replays it, measuring the CPU
overhead reduction vs eager (stream-based) execution. CUDA Graphs eliminate
per-kernel CPU launch overhead (~5–20 µs/kernel) at the cost of reduced flexibility.

## Implementation notes
- Capture: `cudaStreamBeginCapture` → run workload → `cudaStreamEndCapture` → `cudaGraphInstantiate`
- Replay: `cudaGraphLaunch` — near-zero CPU overhead per replay
- Key metric: CPU-side time to launch N kernels (eager vs graph)
- Also measure: latency of first replay vs subsequent replays (JIT compilation cost)
- Limitation: can't change kernel parameters between replays without re-capture

## Results

TODO: run on GPU hardware and fill in this table.

| Scenario | Eager launch time (µs) | Graph replay time (µs) | Speedup |
|----------|------------------------|------------------------|---------|
| 10 small kernels | TODO | TODO | TODO |
| 100 small kernels | TODO | TODO | TODO |
| MLP forward (10 layers) | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU (CUDA 10.0+)
- Build preset: cuda (Linux)
