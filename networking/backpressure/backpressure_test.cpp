// backpressure_test.cpp — two parts:
// (1) TokenBucket: deterministic capacity/refill check, no networking.
// (2) BackpressureSender/Receiver: rank0 floods rank1 with data as fast
//     as credits allow; rank1 processes slowly (simulated) and grants a
//     credit back per unit actually drained. Verifies (a) no data loss,
//     (b) queue depth NEVER exceeds the credit window — a hard bound,
//     true by construction, not an empirically-tuned slack margin (see
//     backpressure.h for why credit-based flow control gives that
//     guarantee where the original threshold-crossing PAUSE/RESUME
//     design could not), (c) the sender actually blocked waiting for
//     credit at least once — proving backpressure engaged, not just "the
//     sender happened to be slow enough anyway."

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
  constexpr size_t kWindow = 20;
  constexpr int kPacketCount = 500;

  auto channels = netcommon::make_tcp_loopback_mesh(2, 35901);
  backpressure::BackpressureSender sender(*channels[0], /*peer=*/1, kWindow);
  backpressure::BackpressureReceiver receiver(*channels[1], /*peer=*/0);
  sender.start();

  std::atomic<size_t> queueDepth{0};
  std::atomic<size_t> maxQueueDepthSeen{0};

  // Receiver: reads packets, enqueues, and a slow "worker" drains them —
  // simulating a receiver that can't keep up. Credit is granted only on
  // drain (grantCredit(), in workerThread below), not on arrival here —
  // that's what couples the sender's rate to real processing capacity.
  std::thread receiverThread([&] {
    for (int i = 0; i < kPacketCount; ++i) {
      uint8_t byte;
      channels[1]->recv(0, &byte, 1);
      size_t depth = queueDepth.fetch_add(1) + 1;
      maxQueueDepthSeen = std::max(maxQueueDepthSeen.load(), depth);
    }
  });
  std::thread workerThread([&] {
    int processed = 0;
    while (processed < kPacketCount) {
      if (queueDepth.load() > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(200)); // slow processing
        queueDepth.fetch_sub(1);
        receiver.grantCredit();
        ++processed;
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
  });

  // Sender: no local TokenBucket needed here — acquireCredit() IS the
  // rate limit now, and unlike TokenBucket's locally-guessed multiple of
  // the receiver's drain rate, it's mechanically coupled to the
  // receiver's actual drain rate (grantCredit() only fires on drain).
  for (int i = 0; i < kPacketCount; ++i) {
    sender.acquireCredit();
    uint8_t byte = 0x42;
    channels[0]->send(1, &byte, 1);
  }

  receiverThread.join();
  workerThread.join();

  // Hard bound, true by construction (see backpressure.h): acquireCredit()
  // cannot let more than kWindow units be sent-but-undrained at once,
  // regardless of scheduling delay — no slack constant needed.
  bool noOverflow = maxQueueDepthSeen.load() <= kWindow;
  bool backpressureEngaged = sender.blockedCount() >= 1;
  bool ok = noOverflow && backpressureEngaged;

  std::printf("Backpressure: max queue depth seen = %zu (window=%zu), sender blocked %zu times: %s\n",
              maxQueueDepthSeen.load(), kWindow, sender.blockedCount(), ok ? "PASS" : "FAIL");
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
