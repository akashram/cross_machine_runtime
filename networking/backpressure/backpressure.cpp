//===- backpressure.cpp - Step 21 implementation --------------------------===//
//
// Control and data deliberately use opposite directions of the same
// full-duplex socket rather than a tagged multiplexed protocol:
// BackpressureReceiver only ever sends PAUSE/RESUME bytes to the sender,
// and the sender only ever sends data to the receiver — TCP's two
// independent directions handle that separation for free, no framing
// needed for the 1-byte control signal.
//
//===----------------------------------------------------------------------===//

#include "backpressure.h"

#include <algorithm>

namespace backpressure {

TokenBucket::TokenBucket(double capacity, double refill_per_sec)
    : capacity_(capacity), refillPerSec_(refill_per_sec), tokens_(capacity),
      lastRefill_(std::chrono::steady_clock::now()) {}

void TokenBucket::refillLocked() {
  auto now = std::chrono::steady_clock::now();
  double elapsedSec = std::chrono::duration<double>(now - lastRefill_).count();
  tokens_ = std::min(capacity_, tokens_ + elapsedSec * refillPerSec_);
  lastRefill_ = now;
}

bool TokenBucket::tryAcquire(double tokens) {
  std::lock_guard<std::mutex> lock(mu_);
  refillLocked();
  if (tokens_ < tokens) return false;
  tokens_ -= tokens;
  return true;
}

double TokenBucket::available() const {
  std::lock_guard<std::mutex> lock(mu_);
  const_cast<TokenBucket *>(this)->refillLocked(); // logically const: refill is bookkeeping, not user-visible state
  return tokens_;
}

namespace {
constexpr uint8_t kResume = 0;
constexpr uint8_t kPause = 1;
} // namespace

BackpressureSender::BackpressureSender(netcommon::Channel &channel, int peer)
    : channel_(channel), peer_(peer) {}

BackpressureSender::~BackpressureSender() { stop(); }

void BackpressureSender::start() {
  running_ = true;
  listenThread_ = std::thread(&BackpressureSender::listenLoop, this);
}

void BackpressureSender::stop() {
  if (!running_.exchange(false)) return;
  if (listenThread_.joinable()) listenThread_.detach(); // see raft.cpp's stop() for why detach, not join
}

void BackpressureSender::listenLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    uint8_t signal;
    try {
      channel_.recv(peer_, &signal, 1);
    } catch (...) {
      return;
    }
    paused_.store(signal == kPause, std::memory_order_relaxed);
  }
}

BackpressureReceiver::BackpressureReceiver(netcommon::Channel &channel, int peer, size_t highWatermark,
                                             size_t lowWatermark)
    : channel_(channel), peer_(peer), highWatermark_(highWatermark), lowWatermark_(lowWatermark) {}

void BackpressureReceiver::reportQueueDepth(size_t depth) {
  std::lock_guard<std::mutex> lock(sendMu_);
  if (!currentlyPaused_ && depth >= highWatermark_) {
    currentlyPaused_ = true;
    uint8_t signal = kPause;
    channel_.send(peer_, &signal, 1);
  } else if (currentlyPaused_ && depth <= lowWatermark_) {
    currentlyPaused_ = false;
    uint8_t signal = kResume;
    channel_.send(peer_, &signal, 1);
  }
}

} // namespace backpressure
