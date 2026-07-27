// batcher_test.cpp — correctness tests for ContinuousBatcher's admission
// logic, independent of continuous_batching_bench.cpp's throughput
// measurement (that file exercises this same class; these tests check
// its behavior in isolation with hand-picked cases).
#include "batcher.h"

#include <cstdio>

using namespace inference_serving;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_admits_up_to_capacity() {
  ContinuousBatcher b;
  for (int i = 0; i < 5; ++i) b.add_request(Request{i, "", 10, 0.0});
  Batch batch = b.next_batch(/*max_batch_size=*/3);
  require(batch.seq_ids.size() == 3, "admits exactly max_batch_size when more are waiting");
  require(b.num_waiting() == 2, "leaves the rest waiting");
  require(b.num_active() == 3, "tracks active count correctly");
}

void test_finish_frees_slot_for_next_admission() {
  ContinuousBatcher b;
  for (int i = 0; i < 3; ++i) b.add_request(Request{i, "", 10, 0.0});
  Batch first = b.next_batch(2);
  require(first.seq_ids.size() == 2, "fills 2 slots first");

  b.finish_sequence(first.seq_ids[0]);
  Batch second = b.next_batch(2);
  require(second.seq_ids.size() == 2, "immediately re-fills the freed slot, not waiting for the whole batch");
  require(b.num_waiting() == 0, "the third request was admitted once a slot freed");
}

void test_empty_batcher_returns_empty_batch() {
  ContinuousBatcher b;
  Batch batch = b.next_batch(4);
  require(batch.seq_ids.empty(), "no requests -> empty batch");
}

void test_finish_unknown_seq_is_noop() {
  ContinuousBatcher b;
  b.add_request(Request{1, "", 5, 0.0});
  b.next_batch(4);
  b.finish_sequence(999);  // never active — must not corrupt state
  require(b.num_active() == 1, "finishing an unknown seq_id leaves active set untouched");
}

} // namespace

int main() {
  test_admits_up_to_capacity();
  test_finish_frees_slot_for_next_admission();
  test_empty_batcher_returns_empty_batch();
  test_finish_unknown_seq_is_noop();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
