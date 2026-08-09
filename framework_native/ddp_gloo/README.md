# ddp_gloo

**Status: code-complete AND locally run — `.venv` (`torch==2.2.2`), real
multi-process.**

## What this measures

PLAN.md Phase 19 step 3: real `torch.nn.parallel.DistributedDataParallel`
training over CPU (`gloo` backend), across REAL separate OS processes
(`torch.multiprocessing` spawn — genuine multi-process, the same
real-process-per-rank standard `distributed_training/training_worker`
(Phase 16) validated, not simulated threads), diffed directly against
`distributed_training/data_parallel`'s own hand-written all-reduce
implementation and its exact task setup.

## Design

- Same task shape as the C++ version: synthetic linear regression
  `y = X.w_true + noise`, a KNOWN true weight vector so convergence is
  checkable against a real target, not just "loss goes down."
  Full-batch (not mini-batch) gradient descent — the point of this step
  is validating distributed MECHANICS, and full-batch keeps the
  single-process-vs-DDP comparison exact, same rationale
  `data_parallel/README.md` gives for its own design.
- 4 ranks, each a REAL separate OS process (`mp.get_context("spawn")`),
  each owning an equal contiguous 1/4 shard — same sharding scheme as the
  C++ version.
- DDP's default gradient all-reduce AVERAGES gradients across ranks. For
  equal-sized shards with per-rank MEAN loss, that average is
  mathematically identical to a single-process full-batch mean-loss
  gradient over the union of all shards — the same equivalence
  `data_parallel/README.md` states for its own sum-then-divide-by-global-
  count approach, here using PyTorch's native average-based DDP instead.
- Compared against a real single-process baseline: same synthetic
  dataset, same initial weights, same number of steps, trained with plain
  (non-distributed) `nn.Linear` + SGD.

## Results (captured 2026-08-09, `torch==2.2.2`, this Mac)

```
  4 REAL separate OS processes (torch.multiprocessing spawn, gloo backend)
  step   baseline_loss   ddp_rank0_shard_loss
     0   4.923969      4.850250
    10   0.659030      0.644030
    20   0.100713      0.098656
    30   0.022947      0.022879
    40   0.011607      0.011635
    50   0.009891      0.009809
    59   0.009632      0.009480

  final single-process loss (full dataset)     = 0.009632
  final DDP global loss (all 400 samples, synced weights) = 0.009622
  max |weight difference| baseline vs. DDP = 0.000000

PASS  DDP's final global loss matches the single-process baseline's to within 1e-3
PASS  DDP's final weights match the single-process baseline's to within 1e-3 (max abs diff)
```

(A benign `gloo` warning — "Unable to resolve hostname to a (local)
address... Using the loopback address as fallback" — prints once per
process on this Mac's networking setup; harmless for a loopback-only
local run, so not filtered out of the captured output above.)

## Findings

- **The final weights match to `0.000000` max absolute difference** —
  essentially exact agreement between 4-process DDP and the single-process
  baseline, not just "in the same ballpark." This directly confirms DDP's
  gradient-averaging semantics really do reduce to the single-process
  full-batch gradient for equal shards, the same mathematical claim
  `distributed_training/data_parallel/README.md` makes and verifies for
  its own hand-written ring all-reduce — here verified independently for
  PyTorch's real, production DDP implementation instead.
- The per-step loss columns above aren't directly comparable row-by-row
  (`ddp_rank0_shard_loss` is rank 0's LOCAL shard loss, a 100-sample
  subset, not the global 400-sample loss the baseline column reports) —
  that's why the actual pass/fail check recomputes DDP's GLOBAL loss
  under the final synchronized weights specifically, rather than
  comparing the printed per-step numbers directly. Documented here so the
  table isn't misread as a mismatch.
- This is a real, separate-process validation, the same standard
  `distributed_training/training_worker` (Phase 16) already established
  for this repo's own hand-written training driver — `torch.multiprocessing`
  spawn creates genuine OS processes communicating over real (loopback)
  `gloo` sockets, not threads simulating ranks the way most of this
  repo's own multi-rank steps do.

## Hardware notes
CPU only, `gloo` backend (no CUDA/NCCL). 4 processes on one Mac; no
multi-machine validation attempted or needed for this step's claim.
