# paged_kv

**Status: code-complete AND locally run — pure CPU bookkeeping, no GPU
dependency. The original stub's "requires GPU" status was too broad; see
Design below.**

## What this measures

PLAN.md Phase 9 step 1: PagedAttention KV cache (Kwon et al., vLLM) —
block table abstraction, logical → physical block mapping, block
allocator with a free list. Validated for correctness: same block
bookkeeping vLLM's design describes, not "same output as non-paged
attention" (this repo has no attention kernel here to compare against —
that comparison belongs to whichever backend's attention kernel actually
reads through this mapping on real K/V memory).

## Design

- `BlockAllocator`: a stack-based free list over `max_blocks` ids —
  `allocate()` pops, `free()` pushes, both O(1). LIFO reuse means
  recently-freed blocks get reused first, the same locality argument
  `foundation/freelist` makes for its allocator.
- `PagedKVCache`: `append_token(seq_id)` grows a sequence's `BlockTable`
  one token at a time, allocating a new physical block only when the
  current logical block is full (`seq_len % block_size == 0`). Block
  table indices are shared across every layer for a given sequence
  (matches vLLM's actual design — all layers need identical
  sequence-length bookkeeping, and nothing requires per-layer physical
  placement to differ for a single-tenant cache).
- **Deliberate deviation from the original stub's signature**: the stub
  had `append(seq_id, layer, KVTensor k, KVTensor v)` — a device pointer
  per call. This file owns block *bookkeeping* only, not K/V bytes: which
  physical block a token lands in is backend-independent (CPU-testable
  today, GPU-usable unchanged later), while actually writing K/V floats
  into that block is a CUDA-kernel concern this repo would implement
  alongside a real attention kernel (out of scope here, same "real code
  where it's meaningful, hardware-gated for the rest" split every other
  phase in this repo uses). `append_token()`'s signature reflects that —
  it advances bookkeeping for one token, returning whether the allocator
  had room, rather than taking tensor data it would have no use for yet.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
OK    block_allocator: free+outstanding == total, no double-issue  300 tests
PASS  proptest: block_allocator invariant
PASS  allocator: exhaustion then reuse after free
PASS  block_table_growth: append_token succeeds within budget  (x5)
PASS  block_table_growth: (block_size+1) tokens use exactly 2 physical blocks
PASS  free_sequence: returns exactly the blocks it held
PASS  free_sequence: forgets the sequence's table
PASS  allocator pressure: a second sequence is blocked when the pool is exhausted
PASS
```

300-trial property test (`foundation/proptest`) over random alloc/free op
sequences: `num_free() + |outstanding blocks| == num_total()` after every
single op, and no block id is ever issued twice while still outstanding —
the exact invariant a double-free or lost-block bug in the allocator
would violate.

## Hardware notes
None for this step's bookkeeping. A real attention kernel that reads K/V
through this mapping (CPU, GPU, FPGA, or TPU) is a separate, backend-
specific concern — see `serving_backend` (step 8) for how this repo routes
between those.
