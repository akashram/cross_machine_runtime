# Adversarial Robustness — Design

Status: Phase 14 code-complete (8/8 steps, including the randomized
smoothing stretch goal, 2026-07-29), all 8 steps **locally run** with real
captured output — no hardware gate anywhere in this phase, same as Phase
13. See each step's own README for full results; this document covers the
decisions behind the split and the tradeoff/transferability findings
measured across steps 4-8.

---

## 1. Why the toy MLP classifier, not the transformer

PLAN.md's "Builds on" line for this phase names `transformer/`, the
autograd engine, and `distributed_training/full_training_loop`. Reading
`full_training_loop/training_loop_test.cpp` settled which of these is the
actual model under attack: it trains `distributed_training::MLP` (a toy
classifier over 2D Gaussian blobs) via the GENERIC reverse-mode tape
(`distributed_training/autograd/autograd.h`), not the transformer. That
tape already wraps its input batch in a `Tensor` (`Tensor
x(shard_x);`) before the forward pass — exactly Goodfellow et al. 2014's
original continuous-feature setting, and the textbook one FGSM/PGD were
designed for. Attacking the transformer's discrete token ids would need a
structurally different approach (perturbing the continuous embedding
lookup instead of the token itself, since tokens can't be perturbed
directly) — a real, separate research question this phase does not take
on. Building on the MLP + generic tape instead meant every one of this
phase's 8 steps needed ZERO modifications to already-relied-upon files
(`autograd.h`, `mlp.h`, `transformer_model.h`) — the lowest-risk path,
and the one PLAN's own "the autograd engine... weight gradients are
already fully implemented" phrasing (step 1) points at directly: the
generic tape never special-cased parameter vs. input Tensors, so
`x.grad()` already worked. This phase's job was making that a tested,
first-class capability, not building a new one.

## 2. Steps 2/3/8 build strictly on step 1, never around it

FGSM (step 2) calls `input_gradient()` once; PGD (step 3) calls it once
per iteration; randomized smoothing's empirical comparison (step 8) calls
`pgd_attack()` unchanged. A real correctness detail this chain forced:
`input_gradient()` must zero the model's weight-parameter grads before
every call, since `Tensor::backward()` only seeds the OUTPUT node and
accumulates (not overwrites) into every ancestor. Without this, PGD's
10 iterations per attack would silently pile up stale weight gradients
across steps — harmless in isolation (attack code never reads them) but
a real hidden footgun for any future caller that reuses the SAME model
Tensors for both attack-crafting and actual training in the same scope
(exactly what step 5's adversarial training loop does, one call after the
other, every epoch).

## 3. PGD generalizes FGSM as a checkable special case, not just a
   conceptual one

`pgd_attack(mlp, x, labels, epsilon, epsilon, 1)` — one full-budget step —
produces BYTE-IDENTICAL output to `fgsm_attack(mlp, x, labels, epsilon)`.
Verified directly in `pgd_test.cpp`, not just argued from the formulas.
This is a strong correctness signal for both functions simultaneously: a
bug that made them diverge at this degenerate configuration would very
likely be a real implementation bug in one or the other.

## 4. The full recall/latency/quality-shaped tradeoff for this phase:
   robustness costs clean accuracy, measured not assumed (steps 4-6)

- **Undefended vulnerability** (step 4): a clean, textbook
  accuracy-collapse curve — 1.000 -> 0.956 -> 0.756 -> 0.544 -> 0.178 ->
  0.011 (FGSM) as epsilon grows from 0 to 4.0 on an undefended model.
  PGD is at least as damaging as FGSM at every epsilon.
- **Adversarial training defends** (step 5): at `epsilon=1.5`, robust
  accuracy goes from 0.722 (undefended) to 0.844 (adversarially trained)
  — the defense measurably works, with no visible clean-accuracy cost YET
  at this one epsilon.
- **Tsipras et al. 2018's tradeoff appears, but only once training
  epsilon is large enough** (step 6): sweeping training epsilon
  `{0.5, 1.5, 2.5, 3.5}`, clean accuracy stays 1.000 at `epsilon=0.5`
  (no cost, attack too weak to matter) but falls to 0.844 then 0.811 as
  training epsilon grows to 2.5 and 3.5 — a real Pareto-frontier-shaped
  relationship (more robustness at a higher epsilon costs more clean
  accuracy), not a fixed property of "adversarial training" as a label.
  This is exactly the shape PLAN.md's step 6 asks to establish, and it
  required the FULL sweep (not step 5's single epsilon) to see.

## 5. Transferability is real but bounded, and the random-noise baseline
   is what makes that measurement mean anything (step 7)

45/90 PGD attacks against model A ({2,16,3}) succeed at `epsilon=2.0`;
17 of those (37.8%) also fool model B ({2,32,16,3}, different width,
depth, and init seed, never queried while crafting the attack). Without
a baseline, "37.8%" alone doesn't establish genuine transfer — it could
just mean a perturbation this large confuses any model. The random-noise
control (same L-infinity magnitude, gradient-free) only fools B 2.2% of
the time — a 17x gap that is the actual evidence this is transfer of the
adversarial DIRECTION found against A, not an artifact of perturbation
size.

## 6. Randomized smoothing's value is the certificate, not necessarily
   beating a targeted defense at its own game (step 8, stretch goal)

Reached as a real implementation (Cohen et al. 2019), not left as an
honest scope note — with one disclosed simplification: a normal (Wald)
approximation confidence bound in place of Cohen et al.'s exact
Clopper-Pearson bound (`inverse_normal_cdf()` itself is exact, via
bisection on `std::erf`; the approximation is specifically in modeling
the vote count as normal rather than binomial/Beta). Measured findings:
every eval point got certified with a substantial average radius (2.974)
— a real consequence of this task's wide class separation (~5.7 units
between centers), not evidence the method never abstains on harder tasks.
Smoothing's empirical gain on PGD-perturbed inputs was modest (0.600 ->
0.617 at `epsilon=2.0`), smaller than adversarial training's gain at a
comparable epsilon (step 6: 0.722 -> 0.844 at `epsilon=1.5`) — consistent
with the structural difference PLAN.md's own framing points at:
smoothing's real selling point is a certificate good against EVERY
attack within its radius, not necessarily outperforming a defense
specifically tuned against the one attack it's being compared on.

## 7. Two real, disclosed scope reductions (not hidden)

- **Confidence bound approximation** (step 8): normal/Wald instead of
  exact Clopper-Pearson — see §6.
- **Toy-scale task throughout**: a 2-feature, 3-class, well-separated
  Gaussian-blob classifier, the same one `full_training_loop` already
  uses. Real findings at this scale (the tradeoff curve's shape, the
  transfer-rate gap over a random baseline) are genuine measurements, not
  fabricated — but magnitudes (e.g. how large a PGD-vs-FGSM gap, how much
  clean-accuracy cost at a given epsilon) are properties of THIS task's
  geometry, not universal constants that would transfer unchanged to a
  higher-dimensional, less-separated real classification problem.
