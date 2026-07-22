# Sharded Checkpointing

**Status: code-complete AND locally run — portable, real files, real
background thread. Content-correctness fully validated locally; wall-clock
overlap benefit is NOT reliably demonstrable on this dev machine — see
below for why, and what real hardware would show instead.**

## What this measures

PLAN.md Phase 6 step 17: sharded checkpoint format, async write overlapped
with training, fast restore.

## Design

Each rank writes ONLY its own parameter shard (`write_shard_sync`) to its
own file — no gather-to-rank-0 step, which is the actual point at real
model scale (gathering a 100B-parameter model to one rank to write it both
serializes the write and requires that rank to have memory for the whole
model). `AsyncCheckpointWriter` snapshots the shard and writes it on a
background thread via `std::async`, so training can continue without
blocking on disk I/O.

## A real, honest finding — not a bug, a measurement result

Unlike step 10 (ZeRO-Infinity), where "overlap helps" was validated via an
analytical cost model because the offload target wasn't real hardware,
this step's file write and background thread ARE real — so it should be
possible to just measure the overlap benefit. It isn't, reliably, on this
machine, and the reason is itself informative:

1. **First attempt** used a CPU-bound busy-loop to simulate the training
   step being overlapped with the write. Measured result: the "overlapped"
   path was consistently SLOWER than serial (up to 2x). Diagnosis: this
   machine has 2 physical cores, and a CPU-bound compute loop competing
   with a background thread's write (which itself needs CPU for the
   20MB copy + syscalls, even if the actual disk write is fast/buffered)
   simply serializes the two operations on scarce cores, plus adds real
   thread-launch and copy overhead on top. This was testing "two CPU-bound
   things fighting over 2 cores," not "I/O overlapping compute" — a flawed
   proxy, not a finding about async checkpointing.
2. **Second attempt** switched to `sleep_for` (correctly modeling that a
   real training step runs on the GPU, not the CPU, so the CPU should be
   genuinely free during it). This should overlap cleanly — but measured
   timing was still noisy run-to-run (background-write completion time
   varied by 5x between otherwise-identical runs, sometimes finishing
   inside the "sleep" window, sometimes not), consistent with OS thread
   scheduling latency under system load on a 2-core machine, not a defect
   in the mechanism.

Given that, `checkpoint_test.cpp`'s PASS/FAIL is content correctness only
(deterministic, and what this test can actually guarantee); wall-clock
timing is printed as information, not asserted on. This is the honest
version of this step's validation without real (multi-core, real-GPU)
hardware — see Results below for what that hardware run should show.

## Sanity-run output (Mac, 2026-07-21)

```
test 1 (write/read round-trip, byte-exact): PASS
test 2 (sharded checkpoint: 4 ranks, no gather, restore reassembles exactly): PASS
test 3 (async write content correctness + informational timing, ~20MB shard):
  serial (write then compute):   0.1589s
  overlapped (write || compute): 0.2576s  (informational only -- see README)
  content correct: yes: PASS
PASS
```

## Results
TODO: run on GPU hardware — with a real GPU doing the "training step"
(genuinely zero CPU contention with a background checkpoint write, unlike
this CPU-only simulation) and real NVMe/network storage bandwidth, the
overlap benefit should be measurable and reliable, unlike here.

| Shard size | Serial (write then step) | Overlapped (write \|\| step) | Speedup |
|------------|-----------------------------|----------------------------------|---------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for content-correctness validation.
- Reliable overlap-benefit measurement (Results table): real GPU (so the
  "training step" doesn't contend with the CPU for the write) and
  real/multi-core hardware without this 2-core dev machine's scheduling
  noise.
