// paged_kv_test.cpp — two layers of testing, matching this repo's
// convention (foundation/test/proptest_test.cpp): a property-based test
// for BlockAllocator's core invariant (no block double-issued while
// outstanding, free+outstanding always equals total), then targeted unit
// tests for PagedKVCache's higher-level block-table bookkeeping.
#include "proptest/proptest.h"

#include "paged_kv_cache.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

using namespace inference_serving;
using proptest::check;
using proptest::gen_vector;
using proptest::gen_int;
using proptest::CheckResult;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// ---------------------------------------------------------------------
// Property: for any sequence of alloc/free-ish ops (encoded as ints —
// even => try to allocate, odd => free the oldest still-outstanding
// block, if any), num_free() + |outstanding| == num_total() after every
// single op, and every outstanding block id is distinct and in-range.
// This is the invariant a free-list allocator must never violate: it's
// exactly what would break under a double-free or a lost-block bug.
// ---------------------------------------------------------------------
bool block_allocator_invariant_holds(const std::vector<int> &ops) {
  constexpr int kNumBlocks = 8;
  BlockAllocator alloc(kNumBlocks);
  std::vector<int> outstanding;  // FIFO of currently-allocated block ids

  for (int op : ops) {
    if (op % 2 == 0) {
      int id = alloc.allocate();
      if (id != -1) {
        if (id < 0 || id >= kNumBlocks) return false;
        if (std::find(outstanding.begin(), outstanding.end(), id) != outstanding.end())
          return false;  // double-issued a still-outstanding block
        outstanding.push_back(id);
      }
    } else if (!outstanding.empty()) {
      int id = outstanding.front();
      outstanding.erase(outstanding.begin());
      alloc.free(id);
    }
    if (alloc.num_free() + static_cast<int>(outstanding.size()) != kNumBlocks) return false;
  }
  return true;
}

void test_block_allocator_property() {
  CheckResult r = check("block_allocator: free+outstanding == total, no double-issue",
                         gen_vector(gen_int(0, 1000), /*max_size=*/200),
                         block_allocator_invariant_holds,
                         [](const std::vector<int> &ops) {
                           std::string s = "[";
                           for (size_t i = 0; i < ops.size() && i < 12; ++i)
                             s += (i ? "," : "") + std::to_string(ops[i]);
                           if (ops.size() > 12) s += ",...";
                           s += "]";
                           return s;
                         },
                         /*num_tests=*/300);
  require(r.passed, "proptest: block_allocator invariant");
}

// ---------------------------------------------------------------------
// Unit tests: BlockAllocator basics.
// ---------------------------------------------------------------------
void test_allocator_exhaustion_and_reuse() {
  BlockAllocator alloc(2);
  int a = alloc.allocate();
  int b = alloc.allocate();
  bool distinct = a != b && a != -1 && b != -1;
  int c = alloc.allocate();
  bool exhausted = (c == -1);
  alloc.free(a);
  int d = alloc.allocate();
  bool reused = (d == a);
  require(distinct && exhausted && reused, "allocator: exhaustion then reuse after free");
}

// ---------------------------------------------------------------------
// Unit tests: PagedKVCache block-table growth and freeing.
// ---------------------------------------------------------------------
void test_block_table_growth() {
  constexpr int kBlockSize = 4;
  PagedKVCache cache(/*num_layers=*/2, /*num_heads=*/2, /*head_dim=*/8, kBlockSize, /*max_blocks=*/16);

  // Appending kBlockSize+1 tokens should grow the block table from 1 to 2
  // physical blocks — the (kBlockSize+1)th token starts a new logical
  // block.
  for (int i = 0; i < kBlockSize + 1; ++i) {
    bool ok = cache.append_token(/*seq_id=*/1);
    require(ok, "block_table_growth: append_token succeeds within budget");
  }
  const BlockTable *table = cache.get_block_table(1);
  require(table != nullptr && table->physical_blocks.size() == 2 && table->seq_len == kBlockSize + 1,
          "block_table_growth: (block_size+1) tokens use exactly 2 physical blocks");
}

void test_free_sequence_returns_blocks() {
  constexpr int kBlockSize = 4;
  PagedKVCache cache(2, 2, 8, kBlockSize, /*max_blocks=*/4);
  for (int i = 0; i < kBlockSize * 3; ++i) cache.append_token(1);  // 3 blocks
  int free_before = cache.num_free_blocks();
  cache.free_sequence(1);
  int free_after = cache.num_free_blocks();
  require(free_after == free_before + 3, "free_sequence: returns exactly the blocks it held");
  require(cache.get_block_table(1) == nullptr, "free_sequence: forgets the sequence's table");
}

void test_allocator_pressure_triggers_failure() {
  constexpr int kBlockSize = 4;
  PagedKVCache cache(1, 1, 4, kBlockSize, /*max_blocks=*/1);  // room for exactly 1 block
  for (int i = 0; i < kBlockSize; ++i) cache.append_token(/*seq_id=*/1);  // fills the only block
  bool second_seq_blocked = !cache.append_token(/*seq_id=*/2);  // no blocks left
  require(second_seq_blocked, "allocator pressure: a second sequence is blocked when the pool is exhausted");
}

}  // namespace

int main() {
  test_block_allocator_property();
  test_allocator_exhaustion_and_reuse();
  test_block_table_growth();
  test_free_sequence_returns_blocks();
  test_allocator_pressure_triggers_failure();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
