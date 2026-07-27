# flash_decoding

**Status: code-complete, hardware-gated — real CUDA kernels, unrun. No
CUDA toolchain on this Mac, same convention as `gpu_engine`.**

## What this measures

PLAN.md Phase 9 step 4: parallelize KV cache access across the sequence
dimension for long contexts, measure latency improvement for 8k+ token
sequences.

## Design

- The problem: `gpu_engine/flash_attn`'s kernel parallelizes over *query*
  tiles (`grid = B*H*ceil(N/Br)`). During autoregressive decode there is
  exactly one query row per (batch, head) — that grid collapses to just
  `B*H` blocks, and with a small decode-serving batch (the common case:
  one request or a handful at a time), `B*H` is nowhere near enough
  blocks to saturate an A100's 108 SMs. Each of those few blocks then
  serially loops the entire KV sequence alone — at 8k+ tokens, most of
  the GPU sits idle while a handful of blocks grind through thousands of
  KV tiles.
- The fix: split the KV dimension itself across additional blocks.
  `flash_decoding_partial` (grid `= (B*H, num_kv_splits)`) computes a
  partial (unnormalized) online-softmax result per KV chunk;
  `flash_decoding_combine` (grid `= B*H`) merges all splits per query
  using the same online-softmax merge rule `flash_attn_fwd`'s running
  update already uses, generalized from pairwise to N-way. This is the
  same algorithm as `flash_attn`, parallelized over the axis that
  actually has work to split at decode time — not a competing technique.
- `choose_num_kv_splits()`: picks enough splits that `B*H*splits`
  approaches the SM count, capped so no split gets fewer than ~256 KV
  rows (below that, per-split fixed overhead — the combine kernel's extra
  read, the second kernel launch — would start to dominate real compute).
- `flash_decoding_bench.cu` compares `num_kv_splits=1` (the decode-time
  degenerate case — equivalent to `flash_attn_fwd`'s grid with `Br=1`)
  against the auto-chosen split count, across N in {512, 2048, 8192,
  16384, 32768} at a deliberately small batch (`B=4, H=8` — 32 total
  (batch,head) pairs, well under a typical GPU's SM count, exactly the
  regime where splitting matters). Correctness: both compute the exact
  same `softmax(QK^T/sqrt(D))V`; only the parallelization strategy
  differs, so any mismatch beyond FP16 reassociation error would be a
  real bug — verified against `splits=1` as the reference, not against a
  separate naive kernel, since they're the same math.

## Results
TODO: run on a GPU instance.

| N | auto splits | splits=1 (ms) | auto (ms) | speedup |
|---|---|---|---|---|
| 512 | TODO | TODO | TODO | TODO |
| 2048 | TODO | TODO | TODO | TODO |
| 8192 | TODO | TODO | TODO | TODO |
| 16384 | TODO | TODO | TODO | TODO |
| 32768 | TODO | TODO | TODO | TODO |

Expected shape: speedup near 1x at small N (not enough work for idle SMs
to matter), growing substantially past 8k tokens where `splits=1` leaves
most SMs idle for this batch size.

## Hardware notes
- Required: NVIDIA GPU, CUDA toolkit (only built when
  `CMAKE_CUDA_COMPILER` is found, mirroring `gpu_engine`'s root
  `CMakeLists.txt` gate).
