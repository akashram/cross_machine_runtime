# tpu_profiler

**Status: code-complete, hardware-gated — real JAX profiler script, unrun.
No TPU device on this Mac.**

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
