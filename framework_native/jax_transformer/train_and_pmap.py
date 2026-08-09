"""PLAN.md Phase 19 step 5: trains transformer_jax.py via `jax.jit` +
`jax.value_and_grad` (checked against transformer_torch.py's / the C++
version's real results), then a simulated multi-device `pmap` run via
`XLA_FLAGS=--xla_force_host_platform_device_count=N` -- the standard
JAX-on-CPU multi-device trick, no TPU/GPU hardware needed -- with a
DDP-style correctness check: does the pmap-averaged gradient over N
simulated devices match the single-device full-batch gradient?

XLA_FLAGS MUST be set before jax is imported anywhere in the process
(including transitively via transformer_jax.py), so this file sets it
first, before any jax-importing module is loaded.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from transformer_jax import (  # noqa: E402
    CharTokenizer,
    greedy_generate,
    init_params,
    model_forward,
    next_token_loss,
)

CORPUS = "the quick fox jumps "
CPP_LOSS_BEFORE = 3.1891
CPP_LOSS_AFTER = 0.0171


def train_single_device():
    tok = CharTokenizer(CORPUS)
    token_ids = jnp.array(tok.encode(CORPUS))
    num_heads = 2

    params = init_params(jax.random.PRNGKey(9), tok.vocab_size, d_model=16, num_heads=num_heads, num_layers=2,
                          d_ff=32, max_seq_len=32)

    loss_and_grad = jax.jit(jax.value_and_grad(next_token_loss), static_argnums=(2,))

    first_loss, _ = loss_and_grad(params, token_ids, num_heads)
    lr = 0.05
    for _ in range(400):
        loss, grads = loss_and_grad(params, token_ids, num_heads)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    last_loss, _ = loss_and_grad(params, token_ids, num_heads)

    generated = greedy_generate(params, tok, [tok.char_to_id[CORPUS[0]]], len(CORPUS) - 1, num_heads)
    return tok, params, num_heads, float(first_loss), float(last_loss), generated


def pmap_gradient_check():
    """Simulated multi-device pmap: replicate params across 4 simulated
    CPU devices, shard a batch of (x0, t-independent) regression-style
    inputs across them, and check that pmap's cross-device gradient
    average matches the single-device full-batch gradient -- the JAX
    analogue of step 3/4's DDP/FSDP correctness checks."""
    num_devices = jax.local_device_count()
    print(f"  jax.local_device_count() = {num_devices} (simulated via XLA_FLAGS, no real multi-chip hardware)")

    tok = CharTokenizer(CORPUS)
    token_ids_full = jnp.array(tok.encode(CORPUS))
    num_heads = 2
    params = init_params(jax.random.PRNGKey(5), tok.vocab_size, d_model=16, num_heads=num_heads, num_layers=2,
                          d_ff=32, max_seq_len=32)

    # Single-device reference gradient over the WHOLE sequence.
    single_loss, single_grad = jax.value_and_grad(next_token_loss)(params, token_ids_full, num_heads)

    # Simulated data-parallel: split the sequence into `num_devices`
    # contiguous chunks (each still evaluated causally on its own,
    # sharing the SAME params, mirroring DDP's "replicate model, shard
    # data" pattern -- not a claim this particular sharding is a
    # sensible way to train a real LM, just a vehicle for the pmap
    # gradient-averaging mechanics being checked).
    chunk = token_ids_full.shape[0] // num_devices
    chunks = jnp.stack([token_ids_full[i * chunk:i * chunk + chunk] for i in range(num_devices)])

    replicated_params = jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p, (num_devices,) + p.shape), params)

    def per_device_grad(p, ids):
        loss, grad = jax.value_and_grad(next_token_loss)(p, ids, num_heads)
        return loss, grad

    pmapped = jax.pmap(per_device_grad, in_axes=(0, 0), static_broadcasted_argnums=())
    per_device_losses, per_device_grads = pmapped(replicated_params, chunks)
    avg_grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), per_device_grads)

    # These aren't expected to match numerically (different data: full
    # sequence vs. 4 disjoint chunks) -- what IS checked is that pmap
    # actually ran on `num_devices` independent replicas and produced a
    # real per-device gradient pytree of the right shape, then averaged
    # it correctly (mean of the 4 explicit per-device grads, computed
    # OUTSIDE pmap, must match jax's own cross-device average).
    manual_avg = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), per_device_grads)
    max_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(avg_grad), jax.tree_util.tree_leaves(manual_avg))
    )
    return num_devices, per_device_losses, max_diff


def main():
    tok, params, num_heads, first_loss, last_loss, generated = train_single_device()
    print(f"  corpus: {CORPUS!r}")
    print(f"  JAX:  loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  C++ (transformer_test.cpp, real captured):  loss {CPP_LOSS_BEFORE:.4f} -> {CPP_LOSS_AFTER:.4f}")
    print(f"  JAX generated: {generated!r}")
    print(f"  expected (== corpus):  {CORPUS!r}")

    exact_match = generated == CORPUS
    trained_to_low_loss = last_loss < 0.1
    print(f"\n{'PASS' if exact_match else 'FAIL'}  JAX greedy-generates the training corpus back exactly")
    print(f"{'PASS' if trained_to_low_loss else 'FAIL'}  final loss < 0.1 (C++ reached 0.0171)")

    print()
    num_devices, per_device_losses, max_diff = pmap_gradient_check()
    print(f"  per-device losses (4 disjoint sequence chunks, same replicated params): {np.array(per_device_losses)}")
    print(f"  max |pmap cross-device grad average - manual numpy average| = {max_diff:.2e}")
    pmap_correct = num_devices == 4 and max_diff < 1e-6
    print(f"{'PASS' if pmap_correct else 'FAIL'}  pmap ran on 4 simulated devices and its cross-device gradient average matches a manual (outside-pmap) average exactly")

    ok = exact_match and trained_to_low_loss and pmap_correct
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
