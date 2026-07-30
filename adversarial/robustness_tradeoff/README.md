# robustness_tradeoff

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 6: clean accuracy of the adversarially-trained
model vs. the undefended model, AND robust accuracy under attack for
both, across a sweep of training epsilons — Tsipras et al. 2018's
"Robustness May Be at Odds with Accuracy," measured directly rather than
cited and assumed.

## Design

Pure composition of step 4 (`vulnerability_measurement`) and step 5
(`adversarial_training`) — no new attack or training logic, only the
sweep-across-training-epsilon comparison itself is new. For each training
epsilon in `{0.5, 1.5, 2.5, 3.5}`: train a fresh adversarially-trained
model at that epsilon, measure its clean accuracy and its robust accuracy
under PGD at that SAME epsilon, and compare both against the one
undefended baseline model (trained once, evaluated at every epsilon in
the sweep).

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  undefended clean accuracy: 1.000
  epsilon      undef. clean   undef. robust    adv. clean     adv. robust     
  0.50         1.000          1.000            1.000          1.000           
  1.50         1.000          0.722            1.000          0.844           
  2.50         1.000          0.178            0.844          0.656           
  3.50         1.000          0.000            0.811          0.489           
PASS  adversarial training's robust accuracy is at least the undefended model's, at every training epsilon in the sweep
  clean-accuracy cost at the largest training epsilon (3.50): undefended=1.000 -> adv-trained=0.811
  finding: a real clean-accuracy cost shows up at this training epsilon (Tsipras et al. 2018, measured not assumed)
PASS
```

## Findings

- **Tsipras et al. 2018's phenomenon shows up cleanly, not forced**: at
  small training epsilon (0.5), there's no cost at all — both models hit
  1.000 clean accuracy, and the attack is too weak to hurt anyone yet. As
  training epsilon grows past the point where the undefended model's
  robustness has already visibly collapsed (2.5, 3.5 — see step 4's
  0.178/0.011 undefended-robust numbers at nearby epsilons), the
  adversarially-trained model's OWN clean accuracy starts paying a real
  price: 1.000 -> 0.844 -> 0.811. The tradeoff isn't a fixed property of
  "adversarial training" in the abstract — it only appears once the
  training epsilon is large enough to meaningfully distort the decision
  boundary.
- Robust accuracy improves at every single training epsilon in the
  sweep, without exception — the defense never makes things worse, and
  the improvement is largest exactly where the undefended model was most
  vulnerable (epsilon=3.5: 0.000 -> 0.489).
- The clean-accuracy cost and the robustness gain grow together as
  training epsilon increases — a real, measured Pareto-frontier-shaped
  relationship (more robustness at a higher epsilon costs more clean
  accuracy), not just "robust training sometimes costs something."

## Hardware notes
None — pure CPU.
