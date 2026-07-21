# Gradient Accumulation

**Status: code-complete AND locally run — portable, builds on
`data_parallel/linreg.h` and `networking/ring_allreduce`.**

## What this measures

PLAN.md Phase 6 step 4: micro-batch accumulation with correct gradient
scaling, and its interaction with data parallelism (the "loss scaling"
PLAN.md mentions is specifically mixed-precision loss scaling, which is
GPU/Phase-3 territory — not applicable here; the scaling bug this step
actually guards against, gradient-accumulation scaling, is device-
independent and is what's tested).

## Design

`GradientAccumulator` (`grad_accum.h`) sums micro-batch gradient sums and
divides by the **total sample count across the accumulation window** —
not the micro-batch count, and not any single micro-batch's size. Test 2
reproduces the bug directly: dividing by micro-batch count (3) instead of
total samples (400) inflates the effective update by 400/3 ≈ 133x. That's
the kind of bug that doesn't crash — it just silently blows up the learning
rate, and would be very easy to ship if the scaling weren't structural.

Test 3 composes accumulation with step 3's data-parallel simulation: 4
ranks, each accumulating two uneven micro-batches (37 + 63 samples) before
converting the correctly-scaled mean back to a sum and `ring_allreduce`-ing
it — must still match the single-process baseline loss curve.

## Sanity-run output (Mac, loopback, 2026-07-21)

```
test 1 (correct scaling matches full-batch): PASS
test 2 (naive scaling inflates the update 133.3x vs correct — expected 133.3x): PASS (bug reproduced as expected)
test 3 (accumulation + data parallel matches baseline): PASS (final loss 2.821568 vs 2.821568)
PASS
```

## Results
TODO: run on GPU hardware — the number that matters here is memory savings
(accumulation lets a logical batch exceed what fits in GPU memory at once)
and any throughput cost from the extra forward/backward passes vs. one
large batch.

| Micro-batch size | Accumulation steps | Peak memory | Throughput vs. single large batch |
|-------------------|--------------------|--------------|-------------------------------------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Memory/throughput validation (Results table): GPU instance.
