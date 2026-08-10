//===- backpressure.h - Token bucket + explicit backpressure signaling ---===//
//
// Two independent, composable primitives. Portable — TokenBucket has no
// network dependency at all; BackpressureChannel is built on
// networking/common::Channel like every other portable Phase 5 step.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "channel.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <thread>

namespace backpressure {

// Classic token bucket: `capacity` tokens max, refilled continuously at
// `refill_per_sec`. tryAcquire is non-blocking — callers needing to wait
// should poll or back off themselves; a blocking variant would need to
// know the caller's tolerance for sleep granularity, which varies too
// much across use cases (RPC admission control vs. bulk transfer
// pacing) to bake into this primitive.
class TokenBucket {
public:
  TokenBucket(double capacity, double refill_per_sec);
  bool tryAcquire(double tokens = 1.0);
  double available() const;

private:
  void refillLocked();

  mutable std::mutex mu_;
  double capacity_;
  double refillPerSec_;
  double tokens_;
  std::chrono::steady_clock::time_point lastRefill_;
};

// Explicit backpressure over a Channel: credit-based flow control, not a
// threshold-crossing PAUSE/RESUME signal. The sender starts with
// `windowSize` credits; every send consumes one; the receiver grants
// exactly one credit back per data unit it actually finishes *processing*
// (not per unit merely received — credit tracks drain capacity, not
// arrival). This is the "explicit backpressure signals between nodes"
// PLAN.md step 21 describes, as opposed to TokenBucket's self-contained
// local rate limiting — the two compose (a sender can rate limit locally
// *and* be credit-limited by a remote receiver).
//
// Why credit-based and not threshold-crossing PAUSE/RESUME (the original
// design here): a PAUSE sent when queue depth crosses a high watermark
// only takes effect once it propagates and the sender notices — under
// real scheduling delay (worse under concurrent system load), the sender
// can keep sending for that entire window, so queue depth overshoot is
// bounded only by (send_rate - drain_rate) * signal_delay, which has no
// fixed ceiling. That surfaced as a real, reproducible flaky test
// (documented in README.md's "real finding") — not fixable by widening a
// slack constant, since the underlying quantity it was bounding is
// genuinely unbounded.
//
// Credit-based flow control has no such gap: acquireCredit() physically
// cannot let the sender put more than `windowSize` units in flight
// without a grant, so max_in_flight <= windowSize is true by
// construction — independent of scheduling delay, exactly the property
// TCP's own receive window, HTTP/2's flow-control window, and gRPC's
// per-stream window all rely on for the same reason.
class BackpressureSender {
public:
  BackpressureSender(netcommon::Channel &channel, int peer, size_t windowSize);
  ~BackpressureSender();

  void start(); // spawns the credit-grant listener

  // Shuts down the read side of `channel_` for `peer` (forcibly
  // unblocking listenLoop()'s in-flight or next recv()) and joins the
  // listener thread -- not detach. See raft.cpp's stop() for the
  // motivating bug (RaftNode::stop() used to detach its receiver
  // threads, reasoning each was "just blocked in recv() forever"; that
  // reasoning has a hole once the owning object can be destructed while
  // a detached thread is still touching it, confirmed via a real
  // SIGSEGV under CPU contention). The original version of this class
  // had the exact same detach-based bug -- caught here by TSan under
  // concurrent load (2026-08-10), not by inspection -- fixed the same
  // way: shutdownPeer() then join(), so by the time stop() returns the
  // listener thread has genuinely exited and channel_ can be safely
  // destroyed.
  void stop();

  // Blocks (busy-polls) until at least one credit is available, then
  // consumes it. Call immediately before sending each data unit — this
  // is what gives the hard in-flight bound described above.
  void acquireCredit();

  // Number of acquireCredit() calls that actually had to wait for a
  // grant (as opposed to a credit already being available). Used by
  // callers/tests to confirm backpressure genuinely engaged, not just
  // "the sender happened to be slow enough on its own."
  size_t blockedCount() const { return blockedCount_.load(std::memory_order_relaxed); }

private:
  void listenLoop(); // increments credits_ on every 1-byte grant received

  netcommon::Channel &channel_;
  int peer_;
  std::atomic<size_t> credits_;
  std::atomic<size_t> blockedCount_{0};
  std::atomic<bool> running_{false};
  std::thread listenThread_;
};

class BackpressureReceiver {
public:
  BackpressureReceiver(netcommon::Channel &channel, int peer);

  // Call after finishing (draining/processing) one data unit — grants
  // exactly one credit back to the sender. Granting on *drain* rather
  // than on *arrival* is what couples the sender's rate to the
  // receiver's real processing capacity instead of its buffering
  // capacity.
  void grantCredit();

private:
  netcommon::Channel &channel_;
  int peer_;
  std::mutex sendMu_;
};

} // namespace backpressure
