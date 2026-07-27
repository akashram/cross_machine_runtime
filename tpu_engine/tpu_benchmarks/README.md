# tpu_benchmarks

**Status: code-complete, hardware-gated — real JAX benchmark scripts,
unrun. No TPU device or multi-chip slice on this Mac.**

## What this measures

PLAN.md Phase 8 step 2: MXU utilization, HBM bandwidth, and ICI
latency/bandwidth vs. EFA — the hardware deep dive that steps 5–9 later
build on.

## Design

- `mxu_util_bench.py`: bf16 matmul across sizes 64..8192, achieved TFLOPS
  reported as a % of published peak (TPU v4: 275 bf16 TFLOPS/chip). Wraps
  the sweep in `jax.profiler.trace()` so a TensorBoard profile is captured
  alongside the top-line number — the *why* behind any low-utilization
  size (padding waste, dispatch overhead, HBM-bound) needs the trace, not
  just the percentage.
- `hbm_bandwidth_bench.py`: large elementwise add (memory-bound, not
  MXU-bound) across 1M–256M element buffers, achieved GB/s as a % of
  published peak (TPU v4: ~1200 GB/s HBM2).
- `ici_latency_bench.py`: `jax.pmap` + `lax.psum` all-reduce across the
  slice's chips, payloads from 4KB to 256MB, latency and achieved
  GB/s per payload size. This is the ICI-vs-EFA comparison PLAN.md asks
  for — psum runs chip-to-chip with no host CPU on the data path, the
  same property `networking/`'s EFA collectives (`ring_allreduce`,
  `halving_doubling`) have relative to a TCP baseline, just one level
  further from the host (dedicated interconnect vs. NIC).

## On the ICI vs. EFA comparison

Neither side of this comparison has a real hardware number in this repo
yet: `networking/`'s EFA-based steps are themselves hardware-gated (no
EFA NIC — Phase 5's status table lists them among the 13 Linux/NIC-gated
steps), and this step has no TPU. Until both get a hardware validation
pass, the comparison in the Results table below is vendor-spec vs.
vendor-spec, not measured vs. measured — noted explicitly rather than
implied as equivalent to a real benchmark-off.

## Results
TODO: run on a GCP TPU v4-8 VM (all three scripts) and, once available,
an EFA-equipped GPU instance (`networking/`'s collectives) for the real
comparison row.

| Metric | TPU v4 (this step) | EFA (networking/, spec) |
|--------|---------------------|--------------------------|
| MXU utilization (large matmul) | TODO | n/a |
| HBM bandwidth | TODO | n/a |
| Interconnect bandwidth (large payload) | TODO | ~100 Gbps/NIC (published EFA spec) |
| Interconnect latency (small payload) | TODO | TODO (networking/ unrun) |

## Hardware notes
- Required: GCP TPU v4-8 VM (multi-chip, for `ici_latency_bench.py`;
  single-chip TPU VM suffices for the MXU/HBM scripts).
- `jax.profiler.trace()` output needs TensorBoard's profile plugin
  (`pip install tensorboard-plugin-profile`) to inspect.
