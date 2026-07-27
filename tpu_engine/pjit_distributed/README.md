# pjit_distributed

**Status: code-complete, hardware-gated — real JAX pjit script, unrun. No
multi-chip TPU slice, and no local JAX install (user declined; see
tpu_engine's step-1 README for the "why write it uncommented anyway"
convention this follows).**

## What this measures

PLAN.md Phase 8 step 7: automatic sharding across TPU chips via `pjit`,
scaling efficiency on a multi-chip slice.

## Design

- `pjit_bench.py` shards a two-layer MLP (up-proj -> relu -> down-proj)
  across a 2D `(data, model)` mesh: up-projection's output feature dim
  split across `model`, down-projection's input (contracting) dim split
  the same way, so no resharding communication happens *between* the two
  matmuls — only the down-projection's output triggers an implicit
  all-reduce over `model`. This mirrors
  `distributed_training/column_parallel_linear` +
  `distributed_training/row_parallel_linear`'s GPU sharding scheme exactly
  (same math, ICI as the interconnect instead of NVLink/EFA), so once both
  sides have real numbers, tensor-parallel scaling efficiency becomes
  directly comparable across backends for the same algorithm.
- Baseline is `model=1` (pure data-parallel, weights replicated) at the
  full chip count — scaling efficiency for each `model` factor is reported
  relative to that baseline's throughput, not relative to a literal
  single-chip run, since the whole slice is in use throughout.
- API note: `jax.experimental.pjit` was folded into plain `jax.jit` in
  newer JAX (an ordinary `jax.jit(in_shardings=..., out_shardings=...)`
  call now does what a separate `pjit` used to). PLAN.md names `pjit`
  explicitly, so this script imports it directly; verify against the JAX
  version `gcp_setup/provision_tpu_vm.sh` installs, and switch to
  `jax.jit` if the standalone import warns or is removed.

## Results
TODO: run on a GCP TPU v4-8 VM (8 chips — enough to sweep model=1/2/4/8).

| data x model | TFLOPS | % of baseline throughput |
|---|---|---|
| 8 x 1 (baseline) | TODO | 100% |
| 4 x 2 | TODO | TODO |
| 2 x 4 | TODO | TODO |
| 1 x 8 | TODO | TODO |

## Hardware notes
- Required: GCP TPU v4-8 VM (8 chips minimum for the full sweep in
  `main()`'s default args).
