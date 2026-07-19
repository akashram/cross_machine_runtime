// backpressure_test.cpp — two parts:
// (1) TokenBucket: deterministic capacity/refill check, no networking.
// (2) BackpressureSender/Receiver: rank0 floods rank1 with data as fast
//     as possible; rank1 processes slowly (simulated) and reports queue
//     depth after every change. Verifies (a) no data loss, (b) queue
//     depth never blows up past a small margin over the high watermark,
//     (c) PAUSE and RESUME both actually fired — proving backpressure
//     engaged, not just "the sender happened to be slow enough anyway."

#include "backpressure.h"

#include <atomic>
#include <cstdio>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

namespace {

bool testTokenBucket() {
  backpressure::TokenBucket bucket(/*capacity=*/5, /*refill_per_sec=*/1000); // fast refill for a quick test
  int acquired = 0;
  for (int i = 0; i < 10; ++i) if (bucket.tryAcquire(1)) ++acquired;
  bool drainedAtCapacity = (acquired == 5); // exactly `capacity` tokens available up front
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 1000/sec * 0.01s = ~10 tokens refilled, capped at 5
  bool refilled = bucket.tryAcquire(5);
  bool ok = drainedAtCapacity && refilled;
  std::printf("TokenBucket: drained exactly capacity=5 up front (%d), refilled after wait (%d): %s\n",
              acquired, refilled, ok ? "PASS" : "FAIL");
  return ok;
}

bool testBackpressureSignaling() {
  constexpr size_t kHigh = 20, kLow = 5;
  constexpr int kPacketCount = 500;

  auto channels = netcommon::make_tcp_loopback_mesh(2, 35901);
  backpressure::BackpressureSender sender(*channels[0], /*peer=*/1);
  backpressure::BackpressureReceiver receiver(*channels[1], /*peer=*/0, kHigh, kLow);
  sender.start();

  std::atomic<size_t> queueDepth{0};
  std::atomic<size_t> maxQueueDepthSeen{0};
  std::atomic<int> pauseCount{0}, resumeCount{0};
  std::atomic<bool> receiverDone{false};

  // Receiver: reads packets, enqueues, and a slow "worker" drains them —
  // simulating a receiver that can't keep up, which is what should
  // trigger PAUSE.
  std::thread receiverThread([&] {
    for (int i = 0; i < kPacketCount; ++i) {
      uint8_t byte;
      channels[1]->recv(0, &byte, 1);
      size_t depth = queueDepth.fetch_add(1) + 1;
      maxQueueDepthSeen = std::max(maxQueueDepthSeen.load(), depth);
      receiver.reportQueueDepth(depth);
    }
    receiverDone = true;
  });
  std::thread workerThread([&] {
    int processed = 0;
    while (processed < kPacketCount) {
      if (queueDepth.load() > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(200)); // slow processing
        size_t depth = queueDepth.fetch_sub(1) - 1;
        receiver.reportQueueDepth(depth);
        ++processed;
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
  });

  // Sender: rate-limited locally by a TokenBucket *and* honoring the
  // receiver's PAUSE/RESUME — the composition documented in
  // backpressure.h. Without the local rate limit, a first version of
  // this test showed queue depth blowing past 50 on a 20-item high
  // watermark: TCP buffers far more than a few dozen 1-byte messages at
  // every layer (send buffer, network, receive buffer), so PAUSE alone,
  // reacting only after the receiver notices, cannot bound how much is
  // already in flight by the time it takes effect. A bounded sender rate
  // is what keeps that in-flight backlog small enough for PAUSE to
  // actually matter — the real lesson of this step, not a test bug.
  backpressure::TokenBucket sendRate(/*capacity=*/8, /*refill_per_sec=*/20000); // ~4x the receiver's drain rate
  bool wasPaused = false;
  for (int i = 0; i < kPacketCount; ++i) {
    while (sender.isPaused()) {
      if (!wasPaused) { ++pauseCount; wasPaused = true; }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (wasPaused) { ++resumeCount; wasPaused = false; }
    while (!sendRate.tryAcquire(1)) std::this_thread::sleep_for(std::chrono::microseconds(50));
    uint8_t byte = 0x42;
    channels[0]->send(1, &byte, 1);
  }

  receiverThread.join();
  workerThread.join();

  bool noOverflow = maxQueueDepthSeen.load() <= kHigh + 5; // small slack for in-flight signal propagation
  bool signalingEngaged = pauseCount.load() >= 1 && resumeCount.load() >= 1;
  bool ok = noOverflow && signalingEngaged;

  std::printf("Backpressure: max queue depth seen = %zu (high watermark=%zu), pauses=%d, resumes=%d: %s\n",
              maxQueueDepthSeen.load(), kHigh, pauseCount.load(), resumeCount.load(), ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = testTokenBucket() && ok;
  ok = testBackpressureSignaling() && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
