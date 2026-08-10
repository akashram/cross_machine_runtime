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

BackpressureSender::BackpressureSender(netcommon::Channel &channel, int peer, size_t windowSize)
    : channel_(channel), peer_(peer), credits_(windowSize) {}

BackpressureSender::~BackpressureSender() { stop(); }

void BackpressureSender::start() {
  running_ = true;
  listenThread_ = std::thread(&BackpressureSender::listenLoop, this);
}

void BackpressureSender::stop() {
  if (!running_.exchange(false)) return;
  channel_.shutdownPeer(peer_); // unblocks listenLoop()'s recv(), see backpressure.h
  if (listenThread_.joinable()) listenThread_.join();
}

void BackpressureSender::listenLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    uint8_t grant;
    try {
      channel_.recv(peer_, &grant, 1);
    } catch (...) {
      return;
    }
    credits_.fetch_add(1, std::memory_order_relaxed);
  }
}

void BackpressureSender::acquireCredit() {
  bool waited = false;
  while (true) {
    size_t cur = credits_.load(std::memory_order_relaxed);
    if (cur > 0 && credits_.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed)) {
      if (waited) blockedCount_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    waited = true;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

BackpressureReceiver::BackpressureReceiver(netcommon::Channel &channel, int peer)
    : channel_(channel), peer_(peer) {}

void BackpressureReceiver::grantCredit() {
  std::lock_guard<std::mutex> lock(sendMu_);
  uint8_t grant = 1;
  channel_.send(peer_, &grant, 1);
}

} // namespace backpressure
