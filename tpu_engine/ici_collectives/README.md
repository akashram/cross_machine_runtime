# ici_collectives

**Status: code-complete, hardware-gated — real JAX script, unrun. No
multi-chip TPU slice, and no local JAX install.**

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
