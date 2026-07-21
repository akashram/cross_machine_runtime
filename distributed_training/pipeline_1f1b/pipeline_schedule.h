#pragma once

// 1F1B pipeline schedule: PLAN.md step 14. `p` stages (one rank per
// stage, real deployments), `m` microbatches. Two schedules are generated
// and simulated here: GPipe (all forwards, then all backwards — simple,
// but holds every in-flight microbatch's activations at once) and 1F1B
// (one-forward-one-backward — same total bubble time as GPipe [this is
// well-established: see the Megatron-LM paper; 1F1B's win is NOT bubble
// reduction], but bounds peak in-flight activations to O(p) instead of
// O(m), since it starts backward as soon as a microbatch's forward is
// far enough along instead of deferring every backward to the end).
//
// This is a real discrete-event simulation of cross-stage dependencies —
// not just the bubble-fraction formula plugged in — because the actual
// engineering risk in a pipeline schedule is getting the DEPENDENCY
// ORDERING right (a stage cannot start microbatch j's backward before its
// own forward for j, or before the next stage's backward for j), and that
// is exactly what a formula can't catch but a simulator with real
// per-stage schedules and cross-stage wait conditions does.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace distributed_training {

struct PipelineOp {
  int microbatch;
  bool is_forward;
};

// Per-stage op list GPipe would run its stages through: F(0..m-1), then
// B(0..m-1) — order within each half does not affect the simulated
// makespan (both respect the same cross-stage dependency structure), only
// which specific activations are alive together, which is what the
// peak-in-flight measurement below is for.
inline std::vector<std::vector<PipelineOp>> gpipe_schedule(int num_stages, int num_microbatches) {
  std::vector<std::vector<PipelineOp>> schedule(static_cast<size_t>(num_stages));
  for (int i = 0; i < num_stages; ++i) {
    for (int j = 0; j < num_microbatches; ++j) schedule[static_cast<size_t>(i)].push_back({j, true});
    for (int j = 0; j < num_microbatches; ++j) schedule[static_cast<size_t>(i)].push_back({j, false});
  }
  return schedule;
}

// Standard 1F1B: stage i (0-indexed from the first stage) runs
// `min(m, p-i-1)` warmup forwards (stages further from the start need
// fewer — they receive their first activation later, by which time
// earlier stages have already run ahead), then alternates one forward
// (of a NEW microbatch) with one backward (of the OLDEST still-pending
// one, FIFO) until every microbatch has been forwarded, then drains the
// remaining pending backwards.
inline std::vector<std::vector<PipelineOp>> one_f_one_b_schedule(int num_stages, int num_microbatches) {
  std::vector<std::vector<PipelineOp>> schedule(static_cast<size_t>(num_stages));
  for (int i = 0; i < num_stages; ++i) {
    int num_warmup = std::min(num_microbatches, num_stages - i - 1);
    int next_forward = 0, next_backward = 0;
    auto &ops = schedule[static_cast<size_t>(i)];
    for (int k = 0; k < num_warmup; ++k) ops.push_back({next_forward++, true});
    int steady_state = num_microbatches - num_warmup;
    for (int k = 0; k < steady_state; ++k) {
      ops.push_back({next_forward++, true});
      ops.push_back({next_backward++, false});
    }
    for (int k = 0; k < num_warmup; ++k) ops.push_back({next_backward++, false});
  }
  return schedule;
}

struct SimulationResult {
  double makespan;
  double bubble_fraction;
  std::vector<int> peak_in_flight_per_stage; // max concurrently-held (forward done, backward not yet) microbatches
};

// Simulates cross-stage dependencies: stage i's forward(j) can't start
// before stage i-1's forward(j) is done; stage i's backward(j) can't
// start before stage i's OWN forward(j) is done, and (if i is not the
// last stage) before stage i+1's backward(j) is done. Each stage
// processes its own op list strictly in order (no reordering) — a fixed-
// point sweep repeatedly advances whichever stage's next op has its
// dependencies satisfied, until every stage's list is exhausted.
inline SimulationResult simulate(const std::vector<std::vector<PipelineOp>> &schedule, double forward_time,
                                  double backward_time) {
  int p = static_cast<int>(schedule.size());
  int m = 0;
  for (auto &ops : schedule)
    for (auto &op : ops) m = std::max(m, op.microbatch + 1);

  constexpr double kUnset = -1.0;
  std::vector<std::vector<double>> forward_done(static_cast<size_t>(p), std::vector<double>(static_cast<size_t>(m), kUnset));
  std::vector<std::vector<double>> backward_done(static_cast<size_t>(p), std::vector<double>(static_cast<size_t>(m), kUnset));
  std::vector<size_t> ptr(static_cast<size_t>(p), 0);
  std::vector<double> stage_free_at(static_cast<size_t>(p), 0.0);
  std::vector<double> busy_time(static_cast<size_t>(p), 0.0);

  bool progress = true;
  int safety = 0;
  while (progress && safety < p * m * 4 + 10) {
    progress = false;
    ++safety;
    for (int i = 0; i < p; ++i) {
      if (ptr[static_cast<size_t>(i)] >= schedule[static_cast<size_t>(i)].size()) continue;
      const PipelineOp &op = schedule[static_cast<size_t>(i)][ptr[static_cast<size_t>(i)]];
      double ready = stage_free_at[static_cast<size_t>(i)];
      bool deps_ready = true;
      if (op.is_forward) {
        if (i > 0) {
          double dep = forward_done[static_cast<size_t>(i - 1)][static_cast<size_t>(op.microbatch)];
          if (dep == kUnset) deps_ready = false; else ready = std::max(ready, dep);
        }
      } else {
        double own_forward = forward_done[static_cast<size_t>(i)][static_cast<size_t>(op.microbatch)];
        if (own_forward == kUnset) deps_ready = false; else ready = std::max(ready, own_forward);
        if (deps_ready && i + 1 < p) {
          double dep = backward_done[static_cast<size_t>(i + 1)][static_cast<size_t>(op.microbatch)];
          if (dep == kUnset) deps_ready = false; else ready = std::max(ready, dep);
        }
      }
      if (!deps_ready) continue;

      double duration = op.is_forward ? forward_time : backward_time;
      double finish = ready + duration;
      if (op.is_forward) forward_done[static_cast<size_t>(i)][static_cast<size_t>(op.microbatch)] = finish;
      else backward_done[static_cast<size_t>(i)][static_cast<size_t>(op.microbatch)] = finish;
      stage_free_at[static_cast<size_t>(i)] = finish;
      busy_time[static_cast<size_t>(i)] += duration;
      ptr[static_cast<size_t>(i)] += 1;
      progress = true;
    }
  }

  double makespan = 0.0;
  for (double t : stage_free_at) makespan = std::max(makespan, t);
  double total_busy = 0.0;
  for (double b : busy_time) total_busy += b;
  double bubble_fraction = 1.0 - total_busy / (static_cast<double>(p) * makespan);

  std::vector<int> peak(static_cast<size_t>(p), 0);
  for (int i = 0; i < p; ++i) {
    std::vector<std::pair<double, int>> events; // (time, delta)
    for (int j = 0; j < m; ++j) {
      events.push_back({forward_done[static_cast<size_t>(i)][static_cast<size_t>(j)], +1});
      events.push_back({backward_done[static_cast<size_t>(i)][static_cast<size_t>(j)], -1});
    }
    std::sort(events.begin(), events.end());
    int cur = 0, peak_i = 0;
    for (auto &e : events) {
      cur += e.second;
      peak_i = std::max(peak_i, cur);
    }
    peak[static_cast<size_t>(i)] = peak_i;
  }

  return SimulationResult{makespan, bubble_fraction, peak};
}

} // namespace distributed_training
