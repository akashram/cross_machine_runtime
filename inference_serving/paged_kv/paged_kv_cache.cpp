#include "paged_kv_cache.h"

#include <cassert>

namespace inference_serving {

BlockAllocator::BlockAllocator(int num_blocks) : num_blocks_(num_blocks) {
  free_list_.reserve(static_cast<std::size_t>(num_blocks));
  // Push in descending order so allocate() (which pops the back) hands
  // out block 0 first — cosmetic (any order is a valid free list), but
  // makes test output/physical ids predictable.
  for (int i = num_blocks - 1; i >= 0; --i) free_list_.push_back(i);
}

int BlockAllocator::allocate() {
  if (free_list_.empty()) return -1;
  int id = free_list_.back();
  free_list_.pop_back();
  return id;
}

void BlockAllocator::free(int block_id) {
  assert(block_id >= 0 && block_id < num_blocks_);
  free_list_.push_back(block_id);
}

PagedKVCache::PagedKVCache(int num_layers, int num_heads, int head_dim, int block_size, int max_blocks)
    : allocator_(max_blocks),
      num_layers_(num_layers),
      num_heads_(num_heads),
      head_dim_(head_dim),
      block_size_(block_size) {}

bool PagedKVCache::append_token(int seq_id) {
  BlockTable &table = seq_tables_[seq_id];  // default-constructs on first use
  bool needs_new_block = table.physical_blocks.empty() || (table.seq_len % block_size_ == 0);
  if (needs_new_block) {
    int block = allocator_.allocate();
    if (block == -1) {
      // Out of blocks. Don't create a dangling empty table entry for a
      // brand-new sequence that never got its first block.
      if (table.physical_blocks.empty() && table.seq_len == 0) seq_tables_.erase(seq_id);
      return false;
    }
    table.physical_blocks.push_back(block);
  }
  table.seq_len += 1;
  return true;
}

const BlockTable *PagedKVCache::get_block_table(int seq_id) const {
  auto it = seq_tables_.find(seq_id);
  return it == seq_tables_.end() ? nullptr : &it->second;
}

void PagedKVCache::free_sequence(int seq_id) {
  auto it = seq_tables_.find(seq_id);
  if (it == seq_tables_.end()) return;
  for (int block : it->second.physical_blocks) allocator_.free(block);
  seq_tables_.erase(it);
}

std::size_t PagedKVCache::bytes_per_block() const {
  // K and V, every layer, every head, head_dim floats per token, block_size tokens.
  return static_cast<std::size_t>(num_layers_) * static_cast<std::size_t>(num_heads_) *
         static_cast<std::size_t>(head_dim_) * static_cast<std::size_t>(block_size_) * 2 * sizeof(float);
}

}  // namespace inference_serving
