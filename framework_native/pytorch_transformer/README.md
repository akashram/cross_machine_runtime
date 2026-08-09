# pytorch_transformer

**Status: code-complete AND locally run — `.venv` (`torch==2.2.2`).**

## What this measures

PLAN.md Phase 19 step 1: a real `torch.nn.Module` reimplementation of
`transformer/transformer_model.h`, architecturally identical (token +
learned positional embedding, N pre-LN decoder blocks with causal
multi-head self-attention + ReLU MLP, final LayerNorm, no-bias linear
output projection), trained on the IDENTICAL corpus, config, and
hyperparameters `transformer/transformer_test.cpp`'s
`test_trains_and_generates()` uses, and diffed against that test's real
captured numbers — the cross-check that the hand-derived C++ backprop and
PyTorch's real autograd agree on the same architecture and task, not just
"a PyTorch transformer trains on something."

## Design

- Same char-level tokenizer ALGORITHM as `transformer/char_tokenizer.h`
  (vocab built in first-occurrence order scanning the corpus left to
  right, not sorted) — a from-scratch Python reimplementation of that
  exact algorithm, not a different tokenizer.
- Same architecture in every structural respect: causal masking via
  `-1e9` before softmax (matching `causal_attention_forward`'s exact
  masking value, not PyTorch's usual `-inf`), ReLU (not GELU) in the MLP,
  pre-LN residual structure, no bias on the output projection.
- Same corpus (`"the quick fox jumps "`), same config (`d_model=16,
  num_heads=2, num_layers=2, d_ff=32, max_seq_len=32`), same training
  regime (400 epochs, `lr=0.05`, plain SGD, no momentum/weight decay) as
  the C++ test — no hyperparameter tuned to make this pass.
- Not expecting bit-identical loss numbers (different RNG streams:
  `torch.manual_seed` vs. `std::mt19937`, different weight-init
  distributions even at "the same" nominal scale) — the real claims
  checked are: comparable convergence (same order-of-magnitude loss
  reduction) and the SAME strict pass/fail bar `transformer_test.cpp`
  itself uses (exact greedy-decode-the-corpus-back).

## Results (captured 2026-08-09, `torch==2.2.2`, this Mac)

```
  corpus: 'the quick fox jumps '
  PyTorch: loss 2.9640 -> 0.0263
  C++ (transformer_test.cpp, real captured):  loss 3.1891 -> 0.0171
  PyTorch generated: 'the quick fox jumps '
  expected (== corpus):  'the quick fox jumps '

PASS  PyTorch greedy-generates the training corpus back exactly, same definition of success as transformer_test.cpp
PASS  final loss < 0.1 (C++ reached 0.0171)
PASS  loss reduction is the same order of magnitude as the C++ run's ~186x (3.1891/0.0171)
```

## Findings

- **The PyTorch port converges to the same qualitative place as the
  hand-derived C++ version** — both start in the `~3.0` range (dominated
  by the random-init cross-entropy over a ~16-character vocabulary,
  `ln(16) ≈ 2.77`, consistent with both), both end near-zero (`0.026` vs.
  `0.017`), and both achieve EXACT greedy-decode-the-corpus-back — the
  strict correctness bar `transformer_test.cpp` itself uses, not a looser
  bar invented for this comparison.
- This is real evidence (not just an architectural description on paper)
  that `transformer_model.h`'s hand-derived backward pass — reusing
  `seq_parallel::layernorm_backward` and
  `tensor_parallel_attn::single_head_attention_backward`, chain-ruled by
  hand through the residual connections — computes the same thing
  PyTorch's autograd computes for the identical forward architecture.
- **A real environment fix was needed before this step could run at
  all**: `torch==2.2.2` (the newest CPU wheel available for this
  platform — Intel/x86_64 macOS, confirmed via `pip index versions
  torch`, capped at `2.2.2` since PyTorch dropped Intel Mac support after
  that release) was compiled against the numpy 1.x ABI, and `.venv` had
  numpy 2.5.1 installed for JAX (`tpu_engine`), breaking
  `torch.from_numpy`. Fixed by pinning `numpy<2` (`1.26.4`) and, as a
  cascading consequence, `scipy<1.14` (`1.13.1`, since `scipy>=1.14`
  requires `numpy>=2.0`) — verified afterward that this did NOT regress
  JAX (`tpu_engine/mxu_opt/mxu_bench.py` re-run clean) before starting
  this step's actual work.

## Hardware notes
CPU only (`torch==2.2.2` CPU wheel, no CUDA/MPS backend used or needed at
this scale).
