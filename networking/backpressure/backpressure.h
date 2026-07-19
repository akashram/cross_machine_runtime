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

// Explicit backpressure over a Channel: the receiving side monitors its
// own queue depth and sends single-byte PAUSE/RESUME control frames back
// to the sender when crossing high/low watermarks; the sending side
// checks a local flag (kept current by a background listener thread)
// before every send. This is the "explicit backpressure signals between
// nodes" PLAN.md step 21 describes, as opposed to TokenBucket's
// self-contained local rate limiting — the two compose (a sender can rate
// limit locally *and* honor a remote PAUSE).
class BackpressureSender {
public:
  BackpressureSender(netcommon::Channel &channel, int peer);
  ~BackpressureSender();

  void start(); // spawns the control-frame listener
  void stop();

  // True if the peer's last signal was PAUSE (or no signal received yet
  // — starts unpaused). Callers should check this before sending data
  // frames on the same channel; this class doesn't intercept data frames
  // itself since the data protocol is application-specific.
  bool isPaused() const { return paused_.load(std::memory_order_relaxed); }

private:
  void listenLoop();

  netcommon::Channel &channel_;
  int peer_;
  std::atomic<bool> paused_{false};
  std::atomic<bool> running_{false};
  std::thread listenThread_;
};

class BackpressureReceiver {
public:
  // `highWatermark`/`lowWatermark` are queue-depth thresholds: crossing
  // high (while not already paused) sends PAUSE; dropping to/below low
  // (while paused) sends RESUME. Hysteresis (two thresholds, not one)
  // avoids rapidly toggling PAUSE/RESUME when depth oscillates around a
  // single value.
  BackpressureReceiver(netcommon::Channel &channel, int peer, size_t highWatermark, size_t lowWatermark);

  // Call whenever the receiver's queue depth changes; sends a control
  // frame if a threshold was crossed.
  void reportQueueDepth(size_t depth);

private:
  netcommon::Channel &channel_;
  int peer_;
  size_t highWatermark_, lowWatermark_;
  bool currentlyPaused_ = false;
  std::mutex sendMu_;
};

} // namespace backpressure
