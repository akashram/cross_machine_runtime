# PPO-based RLHF

**Status: code-complete AND locally run — portable, builds on the real
`transformer/` model, `sft/`'s SFT loss, and `reward_model/`'s reward
model (no CUDA/Linux dependency; multi-rank via simulated ranks over
`networking/ring_allreduce`'s real TCP loopback threads).**

## What this measures

PLAN.md Phase 6 step 24: policy (SFT init) + critic + KL penalty against
a frozen reference model, clipped surrogate objective, reward vs. KL
divergence tradeoff measured across training, monitoring for reward
hacking.

## Design

Setup, reusing prior steps unchanged: `sft_init_policy` SFT-trains the
policy (`sft/sft_trainer.h`'s `masked_next_token_loss`, single-threaded —
gradient sync data-parallelism was already proven in `sft_test.cpp`, so
this step doesn't re-prove it), the SFT'd policy is deep-copied into a
frozen reference model, and `train_reward_model` trains a reward model
(`reward_model/reward_model.h`'s Bradley-Terry loss, also single-threaded)
which is then frozen for the rest of the run. `ppo_rlhf.h` adds a critic
(`CriticParams`, literally `= RewardModelParams` — a value function
`V(prompt)` is architecturally identical to a reward model, just pooled
on prompt-only input and trained by MSE regression instead of
Bradley-Terry), the clipped PPO surrogate objective with its exact
subgradient at the clip boundary, and `rollout_one`.

**Scope**: like `sft/` and `reward_model/`, this domain's response is
always exactly one token, so an "episode" is a single prompt -> one
sampled action -> one terminal reward — a single-step (contextual bandit)
instance of PPO, not a simplification bolted on to dodge multi-token
credit assignment, just what the domain actually is. What's still fully
real: the clipped ratio (including PPO's exact subgradient behavior at
the clip boundary — zero gradient when a step is both clipped AND
selected by the min), the KL penalty folded into the shaped reward via
Schulman's k1 estimator (`old_logprob - ref_logprob`, unbiased since the
sample comes from the old/behavior policy), a learned critic baseline for
the advantage, and multiple PPO epochs over a fixed rollout batch,
data-parallel across 5 simulated ranks each outer iteration (policy and
critic gradients share one `ring_allreduce` call per epoch, concatenated
and split by known length — same trick as `sft`/`reward_model`'s
idx-threading `unflatten_into_grad` overload).

Per outer iteration: roll out one action per prompt from the current
policy (temperature-1.0 softmax over the FULL vocab — no hard restriction
to "sensible" tokens, so policy drift toward nonsense would show up
directly in the reward/KL trace instead of being masked by the action
space), score with the frozen reward model, KL-penalize against the
frozen reference, baseline with the critic — then run `kPpoEpochs` (4)
clipped-surrogate update epochs data-parallel over that fixed batch.

## Sanity-run output (Mac, 2026-07-21)

25 prompts (single-digit addition, `"a+b="`), 2-layer/16-dim policy +
reference + reward model + critic, 10 outer PPO iterations x 4 epochs
each, 5 simulated ranks, clip_eps=0.2, kl_coef=0.1:

```
iter  0: mean_reward=0.928 mean_kl=0.000
iter  1: mean_reward=1.688 mean_kl=0.349
iter  2: mean_reward=2.195 mean_kl=0.285
iter  3: mean_reward=2.681 mean_kl=0.525
iter  4: mean_reward=2.262 mean_kl=0.521
iter  5: mean_reward=2.787 mean_kl=0.793
iter  6: mean_reward=2.244 mean_kl=1.023
iter  7: mean_reward=3.144 mean_kl=0.973
iter  8: mean_reward=2.742 mean_kl=0.978
iter  9: mean_reward=2.898 mean_kl=1.257
reward rose (0.928 -> 2.898) with KL bounded (max |mean KL| = 1.257): PASS
PASS
```

Mean reward rises substantially (0.93 -> ~2.7-3.1, noisy but clearly
trending up) while mean KL divergence against the frozen reference grows
gradually and stays bounded (final iteration 1.26, nowhere near the 3.0
runaway threshold) — exactly the reward-vs-KL tradeoff PLAN.md asks this
step to measure, with no sign of reward hacking (unbounded KL growth
while reward climbs would be the signature; that's not what happens
here). Deterministic across repeated runs (fixed RNG seeds throughout).
TSan-clean (`build/tsan`, ring all-reduce across 5 concurrent rank
threads per PPO epoch).

## Results
TODO: run on GPU hardware — the number that matters at real model/dataset
scale is the reward-vs-KL curve on genuine multi-token responses and a
real preference dataset (this step's toy domain deliberately stays
single-token, matching `sft/`/`reward_model/`'s scope choice), plus
whether the reward-hacking check still holds once responses are long
enough for the policy to find degenerate high-reward outputs the reward
model wasn't trained to penalize.

| Model size | Dataset | Mean reward (iter 0) | Mean reward (final) | Max mean KL |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for correctness/convergence validation.
- Real-scale throughput and dataset validation: GPU instance (see
  CLAUDE.md's hardware-per-phase table for Phase 6).
