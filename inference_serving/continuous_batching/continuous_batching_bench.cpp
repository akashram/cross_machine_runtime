// continuous_batching_bench.cpp — measures throughput improvement from
// continuous batching vs. static batching on the same synthetic request
// trace, using the real ContinuousBatcher class for the continuous side
// (not a hand-derived formula — see batcher.cpp).
//
// PLAN.md Phase 9 step 2: "measure throughput improvement vs static
// batching." Simulated in discrete ticks (one tick = one decode step for
// every active sequence), not wall-clock time — there's no real GPU
// compute happening, so timing this with std::chrono would measure how
// fast this Mac's CPU can run a scheduling loop, not the scheduling
// policy's actual effect. Device-slot-tick accounting is the real
// quantity of interest: how many (tick, slot) units of device time each
// policy spends per useful token generated.
#include "batcher.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace inference_serving;

namespace {

struct SimRequest {
  int id;
  int response_len;   // decode steps needed
  int arrival_tick;
};

std::vector<SimRequest> make_trace(int n, std::mt19937 &rng) {
  // Response lengths drawn from a wide range (1..64) -- the variance
  // that makes static batching's "wait for the longest sequence in the
  // group" cost show up; a uniform-length trace would hide it entirely.
  std::uniform_int_distribution<int> len_dist(1, 64);
  std::uniform_int_distribution<int> arrival_dist(0, n / 4);  // bursty arrivals
  std::vector<SimRequest> trace;
  trace.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    trace.push_back({i, len_dist(rng), arrival_dist(rng)});
  return trace;
}

struct SimResult {
  long long total_ticks = 0;
  long long device_slot_ticks = 0;   // busy (tick, slot) units consumed
  long long useful_token_ticks = 0;  // sum of response lengths -- same for both policies

  double utilization_pct() const { return 100.0 * static_cast<double>(useful_token_ticks) / static_cast<double>(device_slot_ticks); }
  double throughput_tokens_per_tick() const { return static_cast<double>(useful_token_ticks) / static_cast<double>(total_ticks); }
};

// Static batching: requests are grouped into fixed-size batches strictly
// in arrival order (the simplest, most common static-batching policy —
// no reordering). A batch runs for max(response_len) ticks: every slot
// stays occupied — and billed — for the whole batch, even after its own
// sequence finished, because the batch as a whole can't return control
// until the longest sequence completes. That's the real, measurable cost
// this step exists to quantify.
SimResult run_static_batching(std::vector<SimRequest> reqs, int batch_size) {
  SimResult result;
  // Arrival order, ignoring arrival_tick itself (static batching here
  // processes strictly sequentially, one full batch at a time, so a
  // request simply waits for its batch's turn regardless of how early it
  // arrived -- another real cost of static batching this model captures).
  for (std::size_t i = 0; i < reqs.size(); i += static_cast<std::size_t>(batch_size)) {
    std::size_t end = std::min(reqs.size(), i + static_cast<std::size_t>(batch_size));
    int batch_actual_size = static_cast<int>(end - i);
    int max_len = 0;
    long long useful = 0;
    for (std::size_t j = i; j < end; ++j) {
      max_len = std::max(max_len, reqs[j].response_len);
      useful += reqs[j].response_len;
    }
    result.total_ticks += max_len;
    result.device_slot_ticks += static_cast<long long>(batch_actual_size) * max_len;
    result.useful_token_ticks += useful;
  }
  return result;
}

// Continuous batching: drives the real ContinuousBatcher class tick by
// tick. Each tick, next_batch() admits any waiting request into a free
// slot, every active sequence advances one token, and a sequence that
// reaches its response_len calls finish_sequence() -- freeing its slot
// for the very next tick's admission, not the next full-batch boundary.
SimResult run_continuous_batching(std::vector<SimRequest> reqs, int batch_size) {
  ContinuousBatcher batcher;
  std::vector<int> remaining(reqs.size());
  for (const auto &r : reqs) remaining[static_cast<std::size_t>(r.id)] = r.response_len;

  // Sort by arrival tick so add_request() calls happen in the right order.
  std::sort(reqs.begin(), reqs.end(), [](const SimRequest &a, const SimRequest &b) {
    return a.arrival_tick < b.arrival_tick;
  });

  SimResult result;
  for (const auto &r : reqs) result.useful_token_ticks += r.response_len;

  std::size_t next_arrival = 0;
  int tick = 0;
  int finished_count = 0;
  while (finished_count < static_cast<int>(reqs.size())) {
    while (next_arrival < reqs.size() && reqs[next_arrival].arrival_tick <= tick) {
      batcher.add_request(Request{reqs[next_arrival].id, "", reqs[next_arrival].response_len, 0.0});
      ++next_arrival;
    }

    Batch batch = batcher.next_batch(batch_size);
    if (batch.seq_ids.empty() && next_arrival >= reqs.size()) break;  // nothing left, ever

    result.device_slot_ticks += static_cast<long long>(batch.seq_ids.size());
    for (int id : batch.seq_ids) {
      if (--remaining[static_cast<std::size_t>(id)] == 0) {
        batcher.finish_sequence(id);
        ++finished_count;
      }
    }
    ++tick;
  }
  result.total_ticks = tick;
  return result;
}

} // namespace

int main() {
  std::mt19937 rng(42);
  auto trace = make_trace(200, rng);
  constexpr int kBatchSize = 16;

  SimResult static_result = run_static_batching(trace, kBatchSize);
  SimResult cont_result = run_continuous_batching(trace, kBatchSize);

  std::printf("%-20s %14s %18s %14s %12s\n", "policy", "total_ticks", "device_slot_ticks", "util %", "tok/tick");
  std::printf("%-20s %14lld %18lld %13.1f%% %12.2f\n", "static", static_result.total_ticks,
              static_result.device_slot_ticks, static_result.utilization_pct(),
              static_result.throughput_tokens_per_tick());
  std::printf("%-20s %14lld %18lld %13.1f%% %12.2f\n", "continuous", cont_result.total_ticks,
              cont_result.device_slot_ticks, cont_result.utilization_pct(),
              cont_result.throughput_tokens_per_tick());

  double throughput_speedup = cont_result.throughput_tokens_per_tick() / static_result.throughput_tokens_per_tick();
  double makespan_speedup = static_cast<double>(static_result.total_ticks) / static_cast<double>(cont_result.total_ticks);
  std::printf("\nthroughput speedup (tok/tick): %.2fx\n", throughput_speedup);
  std::printf("makespan speedup: %.2fx\n", makespan_speedup);

  return 0;
}
