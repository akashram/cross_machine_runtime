# tpu_benchmarks

**Status: code-complete, hardware-gated — real JAX benchmark scripts, still
unrun on the target TPU hardware. All three scripts were actually run on
CPU (2026-08-01, JAX 0.4.38 in `.venv/`, `ici_latency_bench.py` with
`XLA_FLAGS=--xla_force_host_platform_device_count=4` to get >1 simulated
device for the collective) — see below. None of it is the TPU number this
step needs; it's a toolchain/logic smoke test only.**

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

## Local CPU smoke test (2026-08-01)

`mxu_util_bench.py` (unmodified, full sweep, single CPU device):

| N | TFLOPS | util % (vs TPU v4 peak) |
|---|---|---|
| 64 | 0.02 | 0.0% |
| 128 | 0.03 | 0.0% |
| 256 | 0.09 | 0.0% |
| 512 | 0.13 | 0.0% |
| 1024 | 0.15 | 0.1% |
| 2048 | 0.16 | 0.1% |
| 4096 | 0.14 | 0.1% |
| 8192 | 0.15 | 0.1% |

`hbm_bandwidth_bench.py` (unmodified, full sweep, single CPU device) — a
real, honest finding: achieved GB/s **decreases** as N grows (opposite of
the TPU-HBM "approaches peak at large N" expectation this script's own
docstring states), because at small N the buffers fit in this Mac's CPU
cache (giving an inflated GB/s reading) and at large N it becomes real
DRAM-bandwidth-bound on a 2-physical-core machine — a different memory
hierarchy than TPU HBM, not a bug:

| N (elements) | GB/s | % of TPU v4 HBM peak |
|---|---|---|
| 1,048,576 | 20.8 | 1.7% |
| 4,194,304 | 16.7 | 1.4% |
| 16,777,216 | 15.8 | 1.3% |
| 67,108,864 | 8.6 | 0.7% |
| 268,435,456 | 8.5 | 0.7% |

`ici_latency_bench.py` (`bench_psum_allreduce` called directly for 4
smaller payload sizes, `iters=10`, 4 simulated CPU devices via
`XLA_FLAGS`) — this exercises the real `jax.pmap`+`lax.psum` collective
code path, just time-shared across 4 logical devices on 2 physical cores,
not real chip-to-chip ICI, so these numbers are not remotely comparable to
real interconnect bandwidth:

| payload | latency (us) | GB/s |
|---|---|---|
| 4,096 | 393.5 | 0.02 |
| 65,536 | 269.1 | 0.37 |
| 1,048,576 | 1,184.8 | 1.33 |
| 16,777,216 | 26,521.6 | 0.95 |

All three confirm the scripts and JAX's collective/profiling machinery
work correctly end to end; none of these numbers answer the question this
step actually asks (real TPU MXU/HBM/ICI behavior), which stays TODO below.

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
