#pragma once

// Compute/communication overlap: PLAN.md step 18. While computing layer
// N's gradient, communicate (all-reduce) layer N+1's gradient — real
// backprop's layer-by-layer sequential structure is exactly what makes
// this possible: by the time layer N's gradient is ready, layer N+1's
// all-reduce (kicked off right after it was computed, one step earlier)
// has had a whole layer's worth of compute time to run in the background.
//
// Concurrency design: a SINGLE dedicated communication thread processes
// queued gradients one at a time, all-reducing each over the shared
// Channel. This is deliberate, not incidental — TcpChannel is one socket
// per rank pair (see networking/common/channel.h), not thread-safe for
// concurrent use, so two threads issuing all-reduces on the SAME channel
// at once would interleave bytes on the wire and corrupt the stream. A
// single comm thread draining a queue while the main thread keeps
// computing is both the correct way to avoid that and the realistic
// shape of the real technique (a dedicated communication stream/thread,
// not communication happening wherever compute happens to finish).
// Reuses distributed_training/data_loading's PrefetchQueue as the handoff
// — it was written generically enough (bounded, blocking, producer/
// consumer) that this is exactly the kind of queue it is for.

#include <functional>
#include <thread>
#include <vector>

#include "matrix.h"
#include "../data_loading/prefetch_queue.h"
#include "ring_allreduce.h"

namespace distributed_training {

struct GradTask {
  int layer;
  Matrix grad;
};

// Computes num_layers gradients via compute_fn (called in REVERSE layer
// order, matching real backprop: last layer first), all-reducing each one
// on a background thread as soon as it is produced, overlapped with
// computing the next. Returns the all-reduced gradient for every layer,
// indexed by layer (0..num_layers-1) regardless of computation order.
inline std::vector<Matrix> overlapped_backward(int num_layers, const std::function<Matrix(int)> &compute_fn,
                                                netcommon::Channel &channel) {
  data_loading::PrefetchQueue<GradTask> queue(static_cast<size_t>(num_layers)); // capacity covers every layer: push never blocks
  std::vector<Matrix> results(static_cast<size_t>(num_layers));

  std::thread comm_thread([&]() {
    for (int i = 0; i < num_layers; ++i) {
      auto task = queue.pop();
      Matrix g = task->grad;
      ring_allreduce(g.data(), g.size(), channel);
      results[static_cast<size_t>(task->layer)] = g;
    }
  });

  for (int layer = num_layers - 1; layer >= 0; --layer) {
    Matrix local_grad = compute_fn(layer); // real compute, on the calling thread
    queue.push(GradTask{layer, std::move(local_grad)}); // hand off; comm thread all-reduces concurrently with the NEXT compute_fn call
  }
  comm_thread.join(); // all all-reduces complete and results[] fully written before this returns

  return results;
}

// Naive, non-overlapped baseline: compute then immediately all-reduce,
// one layer at a time, nothing concurrent. Same math, same channel calls
// — used to verify overlap does not change the RESULT, only when the
// communication happens.
inline std::vector<Matrix> serial_backward(int num_layers, const std::function<Matrix(int)> &compute_fn,
                                            netcommon::Channel &channel) {
  std::vector<Matrix> results(static_cast<size_t>(num_layers));
  for (int layer = num_layers - 1; layer >= 0; --layer) {
    Matrix g = compute_fn(layer);
    ring_allreduce(g.data(), g.size(), channel);
    results[static_cast<size_t>(layer)] = g;
  }
  return results;
}

} // namespace distributed_training
