# jax_transformer

**Status: code-complete AND locally run — `.venv` (`jax==0.4.38`).**

## What this measures

PLAN.md Phase 19 step 5: a `jit`/`grad`/`vmap`/`pmap` JAX port of the same
transformer architecture step 1 ported to PyTorch, plus a simulated
multi-device run via `XLA_FLAGS=--xla_force_host_platform_device_count=N`
(the standard JAX-on-CPU multi-device trick) with a DDP-style
cross-device gradient-averaging correctness check.

**Scope note**: pure `jax`/`jax.numpy` (a params pytree — a plain nested
dict of arrays — with manual SGD via `jax.tree_util.tree_map`), no Flax/
Optax — neither is installed (`.venv` only has `jax`/`jaxlib`, already
approved and installed for Phase 8's `tpu_engine`), and pure JAX is
enough to demonstrate the functional-transform model this step is
actually about. A disclosed scope choice.

## Design

- Architecturally identical to `transformer_torch.py`/
  `transformer_model.h`: token+positional embedding, N pre-LN blocks
  (causal multi-head attention + ReLU MLP), final LayerNorm, no-bias
  output projection — same causal-mask value (`-1e9`) as the other two
  implementations.
- Trained via `jax.jit(jax.value_and_grad(next_token_loss))` + a manual
  SGD update (`jax.tree_util.tree_map`), same corpus/config/hyperparameters
  as step 1's PyTorch port and `transformer_test.cpp`'s real run.
- **`XLA_FLAGS` is set before ANY jax-importing module loads** (the first
  line of the entry-point script, before `import jax`) — JAX reads this
  env var once at initialization; setting it later has no effect, a real
  practical gotcha of this trick.
- pmap check: replicate the SAME params across `jax.local_device_count()`
  simulated devices, shard the training sequence into that many disjoint
  chunks, run `jax.pmap` to get one gradient per device, and verify
  `pmap`'s own cross-device average matches a plain (outside-`pmap`)
  numpy-style average computed independently — the JAX analogue of step
  3/4's DDP/FSDP gradient-correctness checks.

## Results (captured 2026-08-09, `jax==0.4.38`, this Mac)

```
  corpus: 'the quick fox jumps '
  JAX:  loss 3.6947 -> 0.0139
  C++ (transformer_test.cpp, real captured):  loss 3.1891 -> 0.0171
  JAX generated: 'the quick fox jumps '
  expected (== corpus):  'the quick fox jumps '

PASS  JAX greedy-generates the training corpus back exactly
PASS  final loss < 0.1 (C++ reached 0.0171)

  jax.local_device_count() = 4 (simulated via XLA_FLAGS, no real multi-chip hardware)
  per-device losses (4 disjoint sequence chunks, same replicated params): [2.5991905 3.4474828 3.9377627 3.7694278]
  max |pmap cross-device grad average - manual numpy average| = 0.00e+00
PASS  pmap ran on 4 simulated devices and its cross-device gradient average matches a manual (outside-pmap) average exactly
```

## Findings

- **A third independent implementation reaches the same qualitative
  result**: JAX converges to `0.0139` (vs. C++'s `0.0171`, PyTorch's
  `0.0263` from step 1 — all comfortably near-zero) and achieves the
  exact same greedy-decode-the-corpus-back correctness bar as both
  earlier implementations. Three independent forward/backward
  implementations (hand-derived C++, PyTorch autograd, JAX's functional
  `grad`) now agree on the same architecture and task.
- **`pmap`'s cross-device gradient average matches a manually-computed
  average EXACTLY** (`0.00e+00` max difference) — real confirmation that
  `jax.pmap`'s implicit cross-device reduction does exactly what it
  claims, checked against an independent computation rather than trusted
  from the API's name.
- `jax.local_device_count()` reporting `4` on a machine with no real
  multi-chip accelerator confirms the `XLA_FLAGS` simulated-device trick
  actually works as documented — a real, verified environment
  configuration, not assumed from JAX's own docs.

## Hardware notes
CPU only. `jax.pmap` here runs across 4 SIMULATED CPU devices (via
`XLA_FLAGS`), not real separate chips/hosts — the mechanics checked
(cross-device gradient averaging) are the same either way, but real
multi-chip scaling behavior is out of scope for this step (see
`tpu_engine/` for the real-TPU-hardware-gated version of that question).
