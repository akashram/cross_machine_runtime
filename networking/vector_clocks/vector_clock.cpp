#include "vector_clock.h"

#include <algorithm>

namespace vclock {

uint64_t LamportClock::tick() { return ++counter_; }

uint64_t LamportClock::onReceive(uint64_t receivedTimestamp) {
  counter_ = std::max(counter_, receivedTimestamp) + 1;
  return counter_;
}

VectorClock::VectorClock(int process_id, int num_processes)
    : processId_(process_id), clock_(static_cast<size_t>(num_processes), 0) {}

const std::vector<uint64_t> &VectorClock::tick() {
  clock_[static_cast<size_t>(processId_)]++;
  return clock_;
}

const std::vector<uint64_t> &VectorClock::onReceive(const std::vector<uint64_t> &received) {
  for (size_t i = 0; i < clock_.size(); ++i) clock_[i] = std::max(clock_[i], received[i]);
  clock_[static_cast<size_t>(processId_)]++;
  return clock_;
}

Order VectorClock::compare(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
  bool aLessEq = true, bLessEq = true;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] > b[i]) aLessEq = false;
    if (b[i] > a[i]) bLessEq = false;
  }
  if (aLessEq && bLessEq) return Order::Equal;
  if (aLessEq) return Order::Before;   // a happened-before b
  if (bLessEq) return Order::After;    // a happened-after b
  return Order::Concurrent;            // neither dominates: no causal relationship
}

} // namespace vclock
