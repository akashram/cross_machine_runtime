"""sparsecore_embedding.py — TPU v5 SparseCore embedding table lookup vs.
a dense-gather embedding baseline (the GPU-equivalent access pattern).

PLAN.md Phase 8 step 13 (TPU v5 only): "implement embedding table lookup
using SparseCore, compare against dense embedding on GPU."

Two distinct code paths here, gated differently:
  - `dense_embedding_lookup`: a plain `jnp.take` gather. This is the
    access pattern a GPU embedding lookup uses (and what
    `distributed_training/`'s steps would use for any embedding table
    there) — no TPU-specific API, runs on any JAX backend including CPU.
    Included here as the baseline this step compares against, not
    duplicated as a separate GPU-only file, since the op itself is
    backend-agnostic; only the *hardware it's timed on* needs to be a GPU
    for a real GPU number.
  - `sparsecore_embedding_lookup`: real SparseCore access from JAX goes
    through the separate `jax-tpu-embedding` library (layered on
    TensorFlow's `TPUEmbeddingV2` API), not a plain `jax.numpy` op — that
    library's API is less stable/documented than core JAX, so this
    function is written against its documented shape (a
    `FeatureConfig`/`TPUEmbedding` setup + an `Optimizer`), and should be
    checked against whatever `jax-tpu-embedding` version is current when
    this actually runs, more so than any other script in this repo.

Unrun here — no TPU v5 (SparseCore only exists from v5 onward, not v4),
no `jax-tpu-embedding` install, and no GPU for the dense-lookup timing.
"""

import time

import jax
import jax.numpy as jnp


def dense_embedding_lookup(table: jnp.ndarray, ids: jnp.ndarray) -> jnp.ndarray:
    """Plain gather — the dense/GPU-equivalent embedding access pattern.
    No TPU-specific API; this is what distributed_training/'s
    `gather`/`scatter` runtime-dialect ops (compiler/dialect/RuntimeOps.td)
    would lower to for an embedding table on CPU/GPU."""
    return jnp.take(table, ids, axis=0)


def bench_dense_lookup(vocab: int, dim: int, batch: int, iters: int = 50) -> float:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    table = jax.random.normal(k1, (vocab, dim), dtype=jnp.bfloat16)
    ids = jax.random.randint(k2, (batch,), 0, vocab)

    f = jax.jit(dense_embedding_lookup)
    f(table, ids).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(table, ids)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    bytes_moved = batch * dim * 2  # bf16, one row gathered per id
    gbps = bytes_moved / elapsed / 1e9
    return elapsed, gbps


def sparsecore_embedding_lookup_sketch(vocab: int, dim: int, batch: int):
    """Sketch of the jax-tpu-embedding-based SparseCore path — not a
    drop-in function call like dense_embedding_lookup, since SparseCore
    embedding tables are configured once (sharded across SparseCore
    instances, with their own optimizer state) and looked up via a
    separate enqueue/dequeue step integrated into the training loop, not
    a single pure function. Written to the documented shape of that API;
    treat exact class/method names as needing a version check, per this
    file's header note.
    """
    # from jax_tpu_embedding import TPUEmbedding, FeatureConfig, TableConfig
    # from jax_tpu_embedding.optimizers import SGD
    #
    # table_config = TableConfig(vocabulary_size=vocab, dim=dim, optimizer=SGD(learning_rate=0.1))
    # feature_config = FeatureConfig(table=table_config, name="embedding")
    # embedding = TPUEmbedding(feature_configs=feature_config, batch_size=batch)
    # embedding.initialize_tables()
    #
    # embedding.enqueue(ids_and_weights)  # dispatches to SparseCore, off the MXU critical path
    # activations = embedding.dequeue()   # gathered embeddings, ready for the dense model body
    raise NotImplementedError(
        "requires jax-tpu-embedding installed against a TPU v5 SparseCore runtime; "
        "see this function's docstring for the documented call shape"
    )


def main(vocab: int = 1_000_000, dim: int = 256, batch: int = 8192) -> None:
    latency, gbps = bench_dense_lookup(vocab, dim, batch)
    print(f"dense gather: vocab={vocab} dim={dim} batch={batch} -> "
          f"{latency * 1e6:.1f} us, {gbps:.2f} GB/s")
    print(
        "\nSparseCore comparison requires TPU v5 + jax-tpu-embedding "
        "(see sparsecore_embedding_lookup_sketch); not runnable here."
    )


if __name__ == "__main__":
    main()
