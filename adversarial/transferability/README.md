# transferability

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 7: do adversarial examples crafted against one
model transfer to fool a differently-sized/differently-trained model?
Real, measured cross-model transfer rate.

## Design

- **Model A**: `{2, 16, 3}` — the same architecture every other step
  uses.
- **Model B**: `{2, 32, 16, 3}` — wider AND one layer deeper, a different
  init seed, trained independently. Adversarial examples are crafted
  against A ONLY; B is never queried during crafting — the entire point
  of a transfer study is that B never sees the attack process, only its
  output.
- **A random-noise baseline, not just a bare transfer-rate number**:
  uniform random perturbation of the SAME L-infinity magnitude as the
  PGD attack, tested against B. If gradient-crafted transfer didn't
  measurably beat this baseline, "transfer" would just mean "any
  perturbation this size confuses any model," not a real property of
  adversarial examples specifically.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  model A {2,16,3} clean accuracy: 1.000
  model B {2,32,16,3} clean accuracy: 1.000
  attacks against A that succeeded: 45/90
  of those, also fooled B (transfer rate): 17/45 = 0.378
PASS  the PGD attack against model A actually succeeds on at least some points (a meaningful transfer measurement needs successful attacks to transfer)
  random noise (same epsilon) fools B: 2/90 = 0.022
PASS  gradient-crafted adversarial examples transfer to model B at a higher rate than same-magnitude random noise fools it (genuine transfer, not just 'any big enough perturbation confuses any model')
```

## Findings

- At `epsilon=2.0`, PGD successfully attacks model A on exactly half the
  evaluation set (45/90) — a deliberately chosen budget where the attack
  is neither trivially weak nor total collapse (consistent with step 4's
  sweep, where `epsilon=2.0` put the undefended model at 0.544 accuracy).
- Of A's 45 successful attacks, 17 (37.8%) ALSO fool model B — a real,
  measured, well-above-chance transfer rate despite B having a
  meaningfully different architecture (twice the width, one more layer)
  and a different random initialization. This is the well-known
  adversarial-transferability phenomenon (Szegedy et al. 2013;
  Papernot et al. 2016), reproduced here as a direct measurement.
- The random-noise baseline (2.2%) is more than 17x smaller than the
  transfer rate (37.8%) — the gap that establishes this as genuine
  transfer of the adversarial DIRECTION found against A, not an artifact
  of the perturbation's magnitude alone. Both models likely learn
  similar (if not identical) local decision-boundary geometry near these
  points, since they're solving the same simple, well-separated task —
  a plausible mechanism for why the gradient direction found against one
  model still points toward the other's boundary too.

## Hardware notes
None — pure CPU.
