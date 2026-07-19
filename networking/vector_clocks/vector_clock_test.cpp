// vector_clock_test.cpp — simulates a 3-process message exchange and
// checks that VectorClock::compare correctly identifies happens-before
// vs. concurrent events, which a scalar Lamport clock provably cannot
// distinguish (also demonstrated below).

#include "vector_clock.h"

#include <cstdio>

int main() {
  int failures = 0;
  auto expect = [&](const char *name, bool cond) {
    std::printf("%-45s %s\n", name, cond ? "OK" : "FAIL");
    if (!cond) ++failures;
  };

  // P0: event A (local tick). P0 sends its clock to P1.
  vclock::VectorClock p0(0, 3), p1(1, 3), p2(2, 3);
  auto eventA = p0.tick();

  // P1: event B — receives P0's message (causally after A).
  auto eventB = p1.onReceive(eventA);

  // P2: event C — purely local, no message exchange with P0 or P1
  // (concurrent with both A and B).
  auto eventC = p2.tick();

  expect("A happened-before B", vclock::VectorClock::compare(eventA, eventB) == vclock::Order::Before);
  expect("B happened-after A", vclock::VectorClock::compare(eventB, eventA) == vclock::Order::After);
  expect("A concurrent with C", vclock::VectorClock::compare(eventA, eventC) == vclock::Order::Concurrent);
  expect("C concurrent with A", vclock::VectorClock::compare(eventC, eventA) == vclock::Order::Concurrent);
  expect("B concurrent with C", vclock::VectorClock::compare(eventB, eventC) == vclock::Order::Concurrent);

  // Lamport clocks: same scenario, but scalar timestamps can't tell A-vs-C
  // (concurrent) apart from A-vs-B (causally ordered) — both just produce
  // "different numbers." This is the vector clock's whole advantage.
  vclock::LamportClock l0, l1, l2;
  uint64_t la = l0.tick();
  uint64_t lb = l1.onReceive(la);
  uint64_t lc = l2.tick();
  expect("Lamport: la < lb (order preserved for causal pair)", la < lb);
  std::printf("(Lamport la=%llu lc=%llu — a real number exists for both, but nothing in\n"
              " a scalar timestamp says whether they're causally related or concurrent;\n"
              " that ambiguity is exactly what the vector clock comparisons above resolve.)\n",
              static_cast<unsigned long long>(la), static_cast<unsigned long long>(lc));

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
