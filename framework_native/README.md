# Phase 19: Framework-Native Training (PyTorch/JAX)

**Status: CODE COMPLETE (6/6 steps), 2026-08-09.** `.venv` (`torch==2.2.2`,
`jax==0.4.38`, `ray==2.49.2`). Every step actually run locally, real
captured output in each step's own README.

## Overview

Scoped 2026-08-09 alongside Phases 17 and 18, closing the one gap in this
repo's Phase 17-19 addition that isn't about a new topic but about
tooling: this repo's entire training stack (autograd, ZeRO-style
sharding, data parallelism) is hand-rolled C++, which doesn't show
fluency with the actual industry-standard frameworks (PyTorch, JAX,
production training frameworks) several of the job descriptions that
motivated Phases 17-19 list as a minimum qualification. Every step here
cross-checks against a from-scratch implementation this repo already
proved correct, rather than building disconnected parallel demos.

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | `pytorch_transformer` | Real `torch.nn.Module` port of `transformer/`, diffed against its real captured numbers |
| 2 | `torch_compile_bench` | `torch.compile` benchmarking — real result is a documented environment gate |
| 3 | `ddp_gloo` | Real `DistributedDataParallel` over CPU `gloo`, 4 real OS processes |
| 4 | `fsdp_vs_zero` | Real `FullyShardedDataParallel` (FULL_SHARD = ZeRO-3), structural comparison to `zero1`/`zero2`/`zero3` |
| 5 | `jax_transformer` | `jit`/`grad`/`vmap`/`pmap` JAX port, simulated 4-device `pmap` correctness check |
| 6 | `production_framework` | DeepSpeed attempted and found genuinely incompatible (two real walls); Ray Train used as the real, working fallback |

## Design highlights

- **Every step cross-checks against something this repo already proved
  correct**: step 1 diffs against `transformer_test.cpp`'s real captured
  loss/generation; step 3 diffs against `distributed_training/
  data_parallel`'s exact task; step 4 diffs against `zero1`/`zero2`/
  `zero3`'s exact correctness methodology; step 5 diffs against BOTH the
  C++ version and step 1's PyTorch port, so three independent
  implementations (hand-derived C++, PyTorch autograd, JAX `grad`) all
  land on the same answer.
- **Real multi-process, not simulated threads, for every distributed
  step** (3, 4, 6): `torch.multiprocessing.spawn`/Ray actors creating
  genuine separate OS processes, the same standard `distributed_training/
  training_worker` (Phase 16) established for this repo's own hand-written
  driver.
- **A real environment-debugging thread runs through the whole phase**:
  step 1 found and fixed a genuine `torch`/`numpy` ABI mismatch before any
  training could happen; step 2 found `torch.compile` is hard-gated by a
  Python-version/torch-version intersection unique to this platform; step
  6 hit that SAME gate again, this time inside DeepSpeed's own source
  code, on top of a separate `distutils`-removal issue — two Phase-19
  steps independently rediscovering the same underlying platform
  constraint from different angles.

## Real findings, not assumed conclusions

- **Three independent implementations agree on the same task** (steps 1,
  5): C++ (loss `3.19->0.017`), PyTorch (`2.96->0.026`), JAX
  (`3.69->0.014`) — all converge to near-zero and all achieve the exact
  same greedy-decode-the-corpus-back correctness bar.
- **`torch.compile` is hard-gated on this platform**, verified by actually
  calling it: `torch==2.2.2` (the newest CPU wheel for Intel Mac) predates
  Dynamo's Python 3.12 support, which only landed in `torch==2.4` (no
  wheel for this platform) — a genuine two-constraint intersection, not a
  benchmark result.
- **DDP's final weights match a single-process baseline to `0.000000`
  max absolute difference** (step 3) — essentially exact agreement across
  4 real separate OS processes.
- **Three real bugs found and fixed in FSDP's setup** (step 4): a
  CUDA-fallback crash on CPU-only builds, a collective-operation hang
  from gating `summon_full_params` behind `if rank==0`, and a
  sharded-storage crash from using the pre-wrap model object outside that
  context — culminating in a real measurement that FSDP shards exactly
  `1/world_size` (`25.0%`) of the parameters per rank, matching ZeRO-3's
  own design target precisely.
- **DeepSpeed genuinely doesn't run here**, confirmed by TWO independent
  real failures (a Python 3.12 stdlib removal, and DeepSpeed's own
  unconditional `torch.compile` usage at import time hitting step 2's
  exact same wall) — not assumed incompatible, not silently skipped.
  Ray Train, the real working fallback, reproduces step 1's numbers
  almost exactly through 2 real distributed worker processes.

See each step's own README for full methodology and captured output;
`framework_native/DESIGN.md` for the phase-level design rationale.

## Hardware notes

CPU only throughout. No GPU/TPU/multi-machine cluster used or needed for
any step's claim.
