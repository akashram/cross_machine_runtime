// hedged_request_test.cpp — simulates a flaky backend (fast most of the
// time, occasionally a straggler) and a reliably-fast second backend.
// Measures p50/p99 latency with vs. without hedging over the same
// request sequence (same random seed) to isolate hedging's effect from
// run-to-run noise.

#include "hedged_request.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

// backend0: 5ms typically, 200ms on ~8% of calls (the straggler).
// backend1: reliably ~6ms.
//
// Takes a seed, not a shared std::mt19937&: hedgedCall() (hedged_request.cpp)
// deliberately lets a "losing" backend thread keep running detached after
// it returns -- no cancellation of stragglers, that's the point of
// hedging -- so a straggler from call i can genuinely still be running
// when call i+1 starts. An earlier version of this test passed a single
// mt19937 by reference into every call's lambda; two overlapping calls'
// threads then touched the same generator with no synchronization, a
// real data race caught by TSan (2026-08-10), not a false positive.
// Constructing a fresh, independently-seeded generator per call removes
// the shared mutable state entirely -- each call's RNG lives only on its
// own thread's stack -- while keeping the property the test actually
// needs: seeding by call index (see main()) makes the no-hedging and
// hedging runs draw the identical straggler pattern call-for-call, same
// as the single-shared-seed version intended.
std::string flakyBackend(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  int ms = (dist(rng) < 0.08) ? 200 : 5;
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  return "ok";
}
std::string reliableBackend() {
  std::this_thread::sleep_for(std::chrono::milliseconds(6));
  return "ok";
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(v.size() * p)];
}

} // namespace

int main() {
  constexpr int kRequests = 200;
  constexpr auto kHedgeDelay = std::chrono::milliseconds(20);

  // Same per-call seed (base + call index) for both runs so backend0's
  // straggler pattern is identical — isolates hedging's effect instead of
  // comparing against different random draws. See flakyBackend() for why
  // this is a fresh generator per call rather than one shared mt19937.
  constexpr unsigned kSeedBase = 42;

  std::vector<double> noHedgeMs, hedgeMs;
  int hedgedCount = 0;

  for (int i = 0; i < kRequests; ++i) {
    auto t0 = Clock::now();
    flakyBackend(kSeedBase + static_cast<unsigned>(i)); // no hedging: always just backend0
    noHedgeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }

  for (int i = 0; i < kRequests; ++i) {
    auto t0 = Clock::now();
    unsigned seed = kSeedBase + static_cast<unsigned>(i);
    auto result = hedging::hedgedCall(
        {[seed] { return flakyBackend(seed); }, [] { return reliableBackend(); }}, kHedgeDelay);
    hedgeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    if (result.wasHedged) ++hedgedCount;
  }

  double noHedgeP50 = percentile(noHedgeMs, 0.50), noHedgeP99 = percentile(noHedgeMs, 0.99);
  double hedgeP50 = percentile(hedgeMs, 0.50), hedgeP99 = percentile(hedgeMs, 0.99);

  std::printf("%-12s %10s %10s\n", "", "p50 (ms)", "p99 (ms)");
  std::printf("%-12s %10.2f %10.2f\n", "no hedging", noHedgeP50, noHedgeP99);
  std::printf("%-12s %10.2f %10.2f\n", "hedged", hedgeP50, hedgeP99);
  std::printf("hedged on %d/%d requests (~%.0f%% straggler rate observed)\n", hedgedCount, kRequests,
              100.0 * hedgedCount / kRequests);

  // The point of hedging: tail latency drops sharply (straggler calls get
  // rescued by the reliable backend ~20ms in), while p50 barely moves
  // (most calls never hedge at all).
  bool p99Improved = hedgeP99 < noHedgeP99 * 0.5; // expect a large win: 200ms stragglers -> ~20-30ms
  bool p50Stable = hedgeP50 < noHedgeP50 * 2.0;   // p50 shouldn't regress much even with hedging overhead
  bool ok = p99Improved && p50Stable;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
