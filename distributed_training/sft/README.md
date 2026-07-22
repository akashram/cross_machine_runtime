# Supervised Fine-Tuning (SFT)

**Status: code-complete AND locally run — portable, builds on the real
`transformer/` model (no CUDA/Linux dependency; multi-rank via simulated
ranks over `networking/ring_allreduce`'s real TCP loopback threads).**

## What this measures

PLAN.md Phase 6 step 22: instruction-response fine-tuning of the real
decoder-only transformer, with per-rank data sharding, loss masking on
prompt tokens, and validated perplexity improvement over the untrained
base checkpoint.

## Design

Task: single-digit addition as instruction tuning (`"2+3="` -> `"5"`),
25 examples (`a,b` in `[0,4]`), sharded 5 examples per rank across 5
simulated data-parallel ranks (`std::async` + real TCP loopback channels,
same harness pattern as `distributed_training/data_parallel`).

`masked_next_token_loss` (`sft_trainer.h`) restricts cross-entropy loss
and its gradient to positions predicting a RESPONSE token — prompt
positions get zero gradient, matching the standard SFT convention (the
model should learn to produce good responses, not memorize prompts).
Per-rank gradients are flattened (`transformer_model.h`'s new
`flatten_grad`/`unflatten_into_grad`/`accumulate_grad`, added this step to
give `ModelGrads` the same all-reduce glue `autograd/mlp.h` already had
for the toy MLP), summed via `ring_allreduce`, then divided by the full
25-example dataset size for a correct mean gradient before the SGD step.

## Sanity-run output (Mac, 2026-07-21)

2-layer, 16-dim transformer, 150 epochs, 5 ranks x 5 examples/rank
(25 examples total), lr=0.05:

```
base model (untrained): perplexity = 18.335
after SFT (150 epochs, 25 examples, 5 ranks): perplexity = 1.069
perplexity improved substantially (< 50% of base): PASS
PASS
```

Perplexity drops from 18.3 (near-random over a 12-character vocabulary)
to 1.07 (near-certain correct digit prediction) — the model reliably
learns the addition task from the masked loss alone. TSan-clean
(`build/tsan`, ring all-reduce across 5 concurrent rank threads).

## Results
TODO: run on GPU hardware — the number that matters at real model/dataset
scale is perplexity improvement on a genuine instruction-tuning dataset
(not toy arithmetic) and wall-clock/throughput of the data-parallel
gradient sync path under real multi-GPU NCCL/EFA instead of loopback TCP.

| Model size | Dataset | Base perplexity | Post-SFT perplexity | Ranks |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for correctness/convergence validation.
- Real-scale throughput and dataset validation: GPU instance (see
  CLAUDE.md's hardware-per-phase table for Phase 6).
