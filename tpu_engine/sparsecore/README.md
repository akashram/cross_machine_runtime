# sparsecore

**Status: code-complete, hardware/toolchain-gated — the SparseCore half
stays unrun (TPU v5 only; SparseCore doesn't exist on v4, a stricter
hardware requirement than the rest of Phase 8). No TPU v5, no
`jax-tpu-embedding` install, no GPU for the dense comparison. The dense
gather half was actually run on CPU (2026-08-01, JAX 0.4.38, `.venv/`) —
see below; it's not the GPU number this step asks for, but it is a real
measurement of real code, unlike the SparseCore sketch which correctly
raises `NotImplementedError`.**

## What this measures

PLAN.md Phase 8 step 13 (v5 only): embedding table lookup using SparseCore,
compared against dense embedding on GPU.

## Design

- `dense_embedding_lookup`: a plain `jnp.take` gather — the backend-agnostic
  access pattern a GPU (or CPU) embedding lookup uses, and what
  `compiler/dialect`'s `runtime.gather` op represents for any backend.
  Timed via `bench_dense_lookup` at 1M-vocab x 256-dim x 8192-batch, a
  representative large-embedding-table shape.
- `sparsecore_embedding_lookup_sketch`: real SparseCore access from JAX
  goes through the separate `jax-tpu-embedding` library (built on
  TensorFlow's `TPUEmbeddingV2` API), not a plain JAX op — a table is
  configured once (`TableConfig`/`FeatureConfig`, with its own optimizer
  state sharded across SparseCore instances) and accessed via an
  enqueue/dequeue step wired into the training loop, structurally
  different from a single pure-function call. Written to that API's
  documented shape and raises `NotImplementedError` rather than faking a
  callable — this library's API surface is less stable than core JAX, so
  pretending it's drop-in callable here would be actively misleading
  about what "code-complete" means for this one step.

## Local CPU smoke test (2026-08-01)

`main()` run unmodified (vocab=1,000,000, dim=256, batch=8192):

| Path | Latency | Bandwidth |
|---|---|---|
| Dense gather (CPU, this Mac) | 823.9 us | 5.09 GB/s |

Confirms the gather-based baseline is real, correct, working code — not
the GPU number PLAN.md step 13 actually asks for as the comparison point,
but a genuine measurement to have on hand once a GPU number exists (a CPU
vs. GPU vs. SparseCore three-way comparison becomes possible with zero
new code, same as this repo's other "portable half already measured"
steps).

## Results
TODO: run `bench_dense_lookup` on a GPU instance for the real dense-GPU
comparison number, and the SparseCore sketch (fleshed out against a current
`jax-tpu-embedding` release) on a TPU v5 VM.

| Path | Latency | Bandwidth |
|---|---|---|
| Dense gather (GPU) | TODO | TODO |
| SparseCore lookup (TPU v5) | TODO | TODO |

## Hardware notes
- Required: TPU v5 VM (SparseCore is v5-only) + `jax-tpu-embedding`
  installed, for the SparseCore path.
- Required: GPU instance, for a real dense-embedding-on-GPU comparison
  number (the dense gather script itself has no TPU/GPU-specific
  dependency and could technically be timed on this Mac's CPU, but that
  wouldn't be the GPU comparison PLAN.md asks for).
