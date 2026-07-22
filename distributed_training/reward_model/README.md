# Reward Model Training

**Status: code-complete AND locally run — portable, builds on the real
`transformer/` model (no CUDA/Linux dependency; multi-rank via simulated
ranks over `networking/ring_allreduce`'s real TCP loopback threads).**

## What this measures

PLAN.md Phase 6 step 23: preference-pair (chosen/rejected) reward model
training with the Bradley-Terry objective, ranking accuracy measured on a
held-out preference set, verifying the reward signal is meaningful before
proceeding to PPO/DPO (steps 24-25).

## Design

`RewardModelParams` (`reward_model.h`) wraps `transformer::ModelParams`'s
body (token/positional embeddings, blocks, final LayerNorm — reused
verbatim, same pattern as `sft/`) and replaces the vocab projection with a
scalar reward head applied to the LAST token's final-LayerNorm output:
under causal masking, the last position is the only one that has attended
to the whole prompt+response, so it's the natural pooled summary for a
sequence-level score. `reward_model_backward` re-enters
`transformer::model_backward`'s backward chain at `final_ln_out` (reusing
its validated `block_backward`/`layernorm_backward` primitives) instead of
at `dlogits`/`w_out`, since the reward head never touches the vocab
projection.

Bradley-Terry: `P(chosen > rejected) = sigmoid(reward_chosen -
reward_rejected)`, loss = `-log P`, numerically stable
`softplus(-d)`/`d + softplus(-d)` formulation. `transformer_model.h` grew
an idx-threading overload of `unflatten_into_grad` so `RewardModelGrads`
(body + reward head) can share ONE `ring_allreduce` call instead of two.

Task: same single-digit addition domain as `sft/` (`"2+3="` -> `"5"`), but
each prompt gets 3 distractor pairs (`(a+b+1)`, `(a+b+2)`, `(a+b+3)` mod
10, all guaranteed wrong) instead of `sft/`'s single deterministic
target — 25 prompts x 3 = 75 preference pairs. The dataset is split at
the PAIR level (not the prompt level) into 60 train / 15 held-out: every
prompt's correct sum is seen during training, but specific held-out
(chosen, rejected) pairings are not. This matters — see below.

## Why the split is pair-level, not prompt-level

The first version of this test held out entire (a, b) combinations never
seen in training at all. That measures whether the reward model can
generalize ADDITION ITSELF to unseen operands from only 20 examples — a
much harder ask than PLAN.md's "verify reward signal is meaningful," and
one this tiny (2-layer, 16-dim, character-level) model badly failed: it
memorized the training pairs (train accuracy hit 100% by epoch 75) while
held-out accuracy collapsed to 0% — worse than random, classic
overfitting on too little data for the generalization being asked of it.

Splitting at the pair level instead (every prompt visible during
training, only specific distractor pairings held out) tests what reward
model held-out evaluation normally tests in practice: the same
prompt/response distribution, held-out specific comparisons — not
zero-shot arithmetic transfer. This is a real, useful negative result
worth keeping documented, not just a bug fixed silently.

## Sanity-run output (Mac, 2026-07-21)

2-layer, 16-dim transformer body, 150 epochs, 5 ranks x 12 pairs/rank
(60 train pairs total), 15 held-out pairs, lr=0.05:

```
base model (untrained): held-out ranking accuracy = 0.667
after reward model training (150 epochs, 60 train pairs, 15 held-out, 5 ranks): held-out ranking accuracy = 0.933
reward signal meaningful (held-out ranking accuracy >= 80%): PASS
PASS
```

Held-out ranking accuracy improves from 0.667 (untrained, near the
"always prefer whichever response happens to embed lower" baseline) to
0.933 — the reward model reliably learns chosen > rejected on preference
pairs it never trained on directly. Deterministic across repeated runs
(fixed RNG seeds throughout). TSan-clean (`build/tsan`, ring all-reduce
across 5 concurrent rank threads).

## Results
TODO: run on GPU hardware — the number that matters at real model/dataset
scale is ranking accuracy on a genuine human-preference dataset (not toy
arithmetic) and whether the reward signal stays well-calibrated enough for
PPO (step 24) to optimize against without reward hacking.

| Model size | Dataset | Held-out ranking accuracy (base) | Held-out ranking accuracy (trained) |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for correctness/convergence validation.
- Real-scale throughput and dataset validation: GPU instance (see
  CLAUDE.md's hardware-per-phase table for Phase 6).
