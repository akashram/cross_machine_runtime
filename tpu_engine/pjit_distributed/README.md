# pjit_distributed

**Status: code-complete, hardware-gated for the real TPU number — still
unrun there. JAX is now installed locally (2026-08-01, `jax==0.4.38` in
`.venv/`, superseding the earlier "no local JAX install" note) and
`pjit_bench.py` was actually run with 4 simulated CPU devices — see below.
Running it for real surfaced and fixed one genuine bug (a `pjit`-compiled
function called outside its `with mesh:` context manager).**

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

## Local CPU smoke test (2026-08-01), and a real bug it found

Running `pjit_bench.py` for the first time (4 simulated CPU devices via
`XLA_FLAGS=--xla_force_host_platform_device_count=4`) crashed immediately:
`main()`'s `bench()` calls invoked the `pjit`-compiled `sharded_mlp`
outside the `with mesh:` block that `sharded_mlp_step()` had already
exited by the time `bench()` ran — this JAX version requires the mesh
context manager to still be active when calling a function compiled with
abstract `PartitionSpec` shardings (not concrete `Sharding` objects).
Fixed by wrapping both `bench()` call sites in `main()` with `with
<mesh>:`. No other logic changed.

With that fix, and much smaller problem sizes than the script's real
defaults (`batch=256, d_model=256, d_ff=512` instead of
4096/4096/16384 — the full default sizes made the 4-simulated-devices-on-
2-physical-cores collective rendezvous extremely slow, since it's real
XLA collective synchronization, not free simulation):

| data x model | latency | % of baseline throughput |
|---|---|---|
| 4 x 1 (baseline) | 1.624 ms/iter | 100% |
| 2 x 2 | 1.823 ms/iter | 89.1% |
| 1 x 4 | 1.727 ms/iter | 94.1% |

These numbers are not meaningful as scaling-efficiency data — all 4
"devices" are simulated processes timesharing 2 real physical cores, not
independent chips with real interconnect, so there's no actual parallel
speedup available to measure either way. What this run does confirm: the
sharding math, mesh construction, and `pjit` compile/execute path are all
genuinely correct (no numerical or shape errors across any mesh
configuration), and the bug above is real and now fixed. The step's actual
deliverable — scaling efficiency across real TPU chips — stays TODO.

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
