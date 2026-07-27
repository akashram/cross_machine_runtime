#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>

// PLAN.md Phase 9 step 1: PagedAttention-style KV cache (Kwon et al.,
// vLLM) — block table abstraction, logical -> physical block mapping,
// block allocator with a free list. This header covers exactly the
// bookkeeping: which physical block a sequence's Nth block of tokens
// lives in, and the free-block pool those physical block ids are drawn
// from. It deliberately does NOT own or move any K/V bytes — vLLM's
// actual attention kernel reads K/V through this mapping on the GPU;
// copying real tensor data is a CUDA-kernel concern (out of scope for
// CPU-portable bookkeeping, same "real code, hardware-gated" split this
// repo uses everywhere else a step needs a GPU). See README.md's Design
// section for why this diverges from the original stub's
// `append(seq_id, layer, KVTensor, KVTensor)` signature.

namespace inference_serving {

// A stack-based free list: allocate() pops, free() pushes. O(1) both
// ways, and LIFO reuse means recently-freed blocks (still warm in the
// device's cache/TLB, in the real GPU-backed version) get reused first —
// the same reuse-locality argument foundation/freelist's allocator makes.
class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks);

  // Returns a free block id, or -1 if the pool is exhausted (caller must
  // preempt/evict a sequence — see sla_scheduler for the preemption
  // policy that would trigger on this).
  int allocate();

  void free(int block_id);

  int num_free() const { return static_cast<int>(free_list_.size()); }
  int num_total() const { return num_blocks_; }

 private:
  std::vector<int> free_list_;
  int num_blocks_;
};

// Logical block index -> physical block id, plus how many tokens this
// sequence has appended so far (needed to know whether the next
// append_token() call fits in the current last block or needs a new
// one).
struct BlockTable {
  std::vector<int> physical_blocks;
  int seq_len = 0;
};

class PagedKVCache {
 public:
  PagedKVCache(int num_layers, int num_heads, int head_dim, int block_size = 16, int max_blocks = 2048);

  // Advance a sequence by one token, allocating a new physical block from
  // the shared pool when the current logical block is full (seq_len %
  // block_size == 0). Block table indices are shared across layers for a
  // given sequence — same design vLLM uses, since every layer's KV for a
  // given logical block index needs the same sequence-length bookkeeping,
  // and per-layer physical placement doesn't need to differ for a
  // single-tenant-per-sequence cache. Returns false (does not advance
  // seq_len) if the allocator is out of blocks.
  bool append_token(int seq_id);

  // nullptr if seq_id has never appended a token.
  const BlockTable *get_block_table(int seq_id) const;

  // Returns every block this sequence holds to the allocator and forgets
  // its table.
  void free_sequence(int seq_id);

  int num_free_blocks() const { return allocator_.num_free(); }
  int block_size() const { return block_size_; }

  // Bytes one physical block occupies across every layer, both K and V,
  // fp32 — the real per-block cost the allocator's `max_blocks` budgets
  // against.
  std::size_t bytes_per_block() const;

 private:
  BlockAllocator allocator_;
  std::unordered_map<int, BlockTable> seq_tables_;
  int num_layers_, num_heads_, head_dim_, block_size_;
};

}  // namespace inference_serving
