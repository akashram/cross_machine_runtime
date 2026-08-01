# tpu_profiler

**Status: code-complete, hardware-gated for the real TPU trace — still
unrun there. Actually run on CPU (2026-08-01, JAX 0.4.38, `.venv/`, 4
simulated devices, much smaller sizes than the real defaults) — see below.
A CPU trace can't show MXU/HBM/ICI bottleneck signatures (none of those
subsystems exist the same way on CPU), so this confirms the capture
mechanism works, not any bottleneck finding.**

## What this measures

PLAN.md Phase 8 step 11: capture TPU profiles, identify MXU utilization
bottlenecks, HBM saturation, ICI contention.

## Design

- `capture_profile.py` wraps `jax.profiler.trace()` around 20 steps of the
  same sharded MLP workload `pjit_distributed`'s step 7 script benchmarks
  (data x model mesh, up-proj -> relu -> down-proj) — one combined trace
  exercising MXU (the matmuls), HBM (activations streaming in/out), and
  ICI (the down-projection's implicit all-reduce over the `model` axis)
  together, rather than another isolated micro-benchmark like steps
  2/6/8's purpose-built scripts.
- Reading the captured trace is inherently a manual/interactive step
  (TensorBoard's profile plugin trace viewer, op_profile tab,
  memory_profile tab) — not something this script automates, since it's a
  UI-driven investigation, not a number to print. The script's final
  printout is a checklist of what to look for and where, tying each of the
  three bottleneck types back to the specific tab and to the earlier steps
  that already predict what they'd show (layout_opt/hbm_sram's ceilings
  for MXU/HBM, hbm_sram's overlap-efficiency framing applied to ICI
  contention specifically).

## Local CPU smoke test (2026-08-01)

`main(trace_dir='/tmp/tpu_profiler_trace', batch=256, d_model=256,
d_ff=512, steps=5)` with 4 simulated CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`, mesh
data=1/model=4): captured 5 steps in 1.009s, trace written successfully to
`/tmp/tpu_profiler_trace`. Confirms `jax.profiler.trace()` plus the
sharded-MLP workload both run correctly together end to end (same pjit
sharding code as `pjit_distributed`, same mesh-context fix applies since
this script already had the `pjit` call correctly inside `with mesh:`).
Trace wasn't opened in TensorBoard (no `tensorboard-plugin-profile`
installed) — and even if it were, a CPU trace has no MXU/HBM/ICI signal to
find, since none of those TPU-specific subsystems are being exercised.
The step's actual deliverable (real bottleneck findings from a real TPU
trace) stays TODO.

## Results
TODO: run on a GCP TPU v4-8 VM, open the trace in TensorBoard, and fill in
findings from each tab.

| Bottleneck | Tab | Finding |
|---|---|---|
| MXU utilization | op_profile | TODO |
| HBM saturation | memory_profile | TODO |
| ICI contention | trace_viewer | TODO |

## Hardware notes
- Required: GCP TPU v4-8 VM (multi-chip, to actually exercise ICI
  contention via the sharded workload's all-reduce).
- Reading the trace needs `pip install tensorboard-plugin-profile` locally
  (or on a machine with network access to the TPU VM's trace directory).
