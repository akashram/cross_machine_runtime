# ici_collectives

**Status: code-complete, hardware-gated for the real TPU number — still
unrun there. JAX is now installed locally (2026-08-01, `jax==0.4.38`) and
`bench_grad_allreduce` was run directly for the smallest (125M-param)
class with 2 simulated CPU devices — see below. The 1.3B/7B classes were
skipped: this Mac has 8GB total RAM, and those classes need 5-14GB+ per
simulated device, which real TPU HBM has and this Mac doesn't.**

## What this measures

PLAN.md Phase 8 step 8: gradient all-reduce over ICI, measured against EFA
all-reduce on an equivalent GPU setup.

## Design

- `ici_grad_allreduce.py` applies `tpu_benchmarks/ici_latency_bench.py`'s
  (step 2) raw-bandwidth `pmap`+`psum` methodology to realistically-shaped
  gradient payloads (125M/1.3B/7B-parameter-class flattened gradients,
  bf16) instead of a synthetic byte sweep — the actual collective a
  data-parallel training step issues, matching what
  `distributed_training/`'s GPU steps do via
  `networking/ring_allreduce`/`collectives` over EFA/TCP, just over ICI.
- Deliberately separate from step 2's script rather than a shared helper:
  step 2 characterizes the interconnect itself (arbitrary sizes, isolating
  hardware behavior), this step characterizes the training workload's
  actual collective traffic — different questions, worth keeping visually
  distinct even though the underlying `psum` call is nearly identical.

## On the ICI vs. EFA comparison

The closest *real* (not spec) number in this repo is
`networking/ring_allreduce`'s ~0.1 GB/s effective bandwidth — but that
run is loopback TCP through simulated-rank threads on this Mac, and its
own README says explicitly that number is overhead-dominated, not
representative of real interconnect throughput. It's a documented floor,
not a fair comparison point. A real ICI-vs-EFA comparison needs this step
run on a TPU slice *and* `networking/`'s EFA steps run on real
EFA-equipped nodes — both hardware-gated, neither done yet.

## Local CPU smoke test (2026-08-01)

`bench_grad_allreduce(125_000_000, iters=10)` called directly with 2
simulated CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=2`):

| Model class | Params | Latency | GB/s |
|---|---|---|---|
| 125M-param-class | 125M | 1216.9 ms | 0.21 |

Two simulated devices timesharing this Mac's 2 physical cores is nowhere
near real chip-to-chip ICI, so this latency/bandwidth number is not
comparable to the TPU number this step needs, or even to
`ici_latency_bench.py`'s own CPU smoke-test numbers (different payload
shape, different device count). What it does confirm: the `pmap`+`psum`
gradient-allreduce code path executes correctly end to end on a real
(if tiny) multi-device payload. 1.3B/7B-param classes are not locally
runnable at all — even 2 devices at that size exceeds this Mac's 8GB RAM,
which is itself a real, disclosed limitation, not a bug.

## Results
TODO: run on a GCP TPU v4-8 VM.

| Model class | Params | Latency (ms) | GB/s |
|---|---|---|---|
| 125M-param-class | 125M | TODO | TODO |
| 1.3B-param-class | 1.3B | TODO | TODO |
| 7B-param-class | 7B | TODO | TODO |
| EFA equivalent (networking/, real hardware) | — | TODO | TODO |

## Hardware notes
- Required: GCP TPU v4-8 VM.
- Fair EFA comparison additionally needs an EFA-equipped multi-node GPU
  setup for `networking/ring_allreduce`/`halving_doubling` — see Phase 5's
  hardware-gated status table.
