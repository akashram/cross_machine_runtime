//===- vector_clock.h - Lamport timestamps + vector clocks --------------===//
//
// Portable — pure logic, no network dependency (network dependency is
// the *caller's* job: piggyback a clock on every message this project's
// Channel-based transport sends, à la Raft's AppendEntries or Chandy-
// Lamport's markers). This is the building block chandy_lamport (step 18)
// uses for "did this snapshot see a consistent cut."
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <vector>

namespace vclock {

// Lamport scalar timestamp: strictly weaker than a vector clock (can say
// "this happened before that" is *possible* but can't distinguish
// "definitely concurrent" from "definitely ordered" the way a vector
// clock can) — included because it's the simpler primitive a vector
// clock generalizes, and some callers (a simple audit log) only need
// total-order-consistent timestamps, not full causality tracking.
class LamportClock {
public:
  uint64_t tick(); // local event: increment and return new value
  uint64_t onReceive(uint64_t receivedTimestamp); // message arrival: max(local, received)+1
  uint64_t value() const { return counter_; }

private:
  uint64_t counter_ = 0;
};

enum class Order { Before, After, Concurrent, Equal };

// One component per process. process_id indexes which component this
// instance increments on a local event.
class VectorClock {
public:
  explicit VectorClock(int process_id, int num_processes);

  const std::vector<uint64_t> &tick();            // local event
  const std::vector<uint64_t> &onReceive(const std::vector<uint64_t> &received);
  const std::vector<uint64_t> &values() const { return clock_; }

  // Compares two *already-captured* snapshots (not two live clocks) —
  // the caller captures values() at the point of interest (e.g. when
  // sending/logging an event) since a live VectorClock keeps changing.
  static Order compare(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b);

private:
  int processId_;
  std::vector<uint64_t> clock_;
};

} // namespace vclock
