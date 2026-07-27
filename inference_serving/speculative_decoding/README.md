# speculative_decoding

**Status: code-complete AND locally run — real draft + verifier
transformers, both actually trained and run on this Mac's CPU (reusing
`transformer/`), not stubbed or GPU-gated.**

## What this measures

PLAN.md Phase 9 step 5: small draft model + large verifier, token
acceptance/rejection, acceptance rate measurement, throughput vs. latency
tradeoff.

## Design

- **Deliberate deviation from the original stub's constructor**: the stub
  took `draft_model_path`/`verifier_model_path` strings (load checkpoints
  from disk). This repo has no 1B/7B checkpoints to load — instead,
  `SpecDecoder` takes two real, in-process-trained `transformer::ModelParams`
  (a small draft config and a larger verifier config, both trained on the
  same corpus in `spec_decode_test.cpp`, the same training loop
  `transformer/transformer_test.cpp` already validates). Smaller-vs-larger
  config of the same real architecture stands in for 1B-vs-7B — the
  algorithm doesn't care about absolute parameter count, only that the
  draft is cheap and the verifier is authoritative.
- **Acceptance rule: greedy-equivalence.** Chen et al.'s algorithm verifies
  with rejection *sampling* against a stochastic draft distribution; this
  repo's draft and verifier are both used greedily (deterministic argmax),
  so a proposed token is accepted iff the verifier's own argmax at that
  position matches it — the well-defined special case of rejection
  sampling when both distributions collapse to point masses.
- `propose()`: draft model, run token-by-token (cheap, that's the point),
  greedy-decodes `draft_len` tokens.
- `verify_full()` (private, shared by public `verify()` and
  `generate_round()`): ONE verifier forward pass over
  `prompt + proposed_tokens`. Finds the first position where the
  verifier's argmax disagrees with the proposal; also extracts the
  **bonus token** — the verifier's own argmax at that same disagreement
  position (or one past the end, if every proposed token was accepted) —
  since that forward pass already computed it for free. This is the real
  algorithm's actual throughput mechanism: every round produces at least
  1 new token even on total rejection, and up to `draft_len + 1` on full
  acceptance.
- **Correctness property this repo actually tests**, not just "it runs":
  speculative decoding's output must be *exactly* what greedy-decoding the
  verifier alone would produce — draft quality only changes how many
  verifier calls that takes, never the result. `spec_decode_test.cpp`
  checks this literally: generates the same corpus continuation via
  `SpecDecoder` and via plain verifier-only greedy decoding, and asserts
  the token sequences are identical, for two very different draft models
  (untrained/random and meaningfully trained).

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  precondition: well-trained verifier greedy-reproduces the corpus
PASS  untrained draft: spec-decoded output exactly matches verifier-alone greedy output
  [untrained draft] acceptance rate=0.012  verifier calls: spec=41 naive=43  speedup=1.05x
PASS  trained draft: spec-decoded output exactly matches verifier-alone greedy output
  [trained draft] acceptance rate=0.972  verifier calls: spec=9 naive=43  speedup=4.78x
PASS
```

## Findings

- **Correctness holds regardless of draft quality**: both the untrained
  (essentially random, 1.2% acceptance) and trained (97.2% acceptance)
  draft models produce token-for-token identical output to plain
  verifier-only greedy decoding. This is the property that makes
  speculative decoding a strict speedup with no quality tradeoff, not an
  approximation — verified here, not just claimed.
- **Draft quality is the entire lever on speedup**: verifier-call count
  drops from 41/43 (untrained draft — barely better than naive, since a
  near-random draft gets rejected almost every position, but the bonus
  token still guarantees *some* progress every round) to 9/43 (trained
  draft, 97.2% acceptance) — a 4.78x reduction in verifier forward passes
  for generating the same 43-token continuation. In a real deployment,
  verifier forward passes (the large model) dominate cost, so this ratio
  is a direct proxy for wall-clock speedup, not just a call-count
  curiosity.
- Even the near-random draft case never *hurts* — 41 verifier calls vs.
  naive's 43 — because the bonus-token mechanism means a round can never
  produce zero new tokens, only exactly 1 (full rejection) up to
  `draft_len + 1` (full acceptance). This lower bound is structural, not
  a property of this specific corpus.

## Hardware notes
None. Both models are the real `transformer/` architecture, trained and
run entirely on CPU; a production deployment would swap in real 1B/7B
checkpoints on GPU without changing `SpecDecoder`'s algorithm.
