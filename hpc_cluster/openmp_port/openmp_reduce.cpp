// openmp_reduce.cpp -- PLAN.md Phase 22 step 2: port a hand-threaded
// kernel to OpenMP pragmas, compared directly against the hand-rolled
// version for correctness and measured speedup.
//
// Kernel chosen: parallel reduction (sum) over a float array -- the
// textbook case OpenMP's `reduction()` clause exists for, and directly
// comparable to foundation::WorkStealingPool::parallel_for's own
// shard-then-combine pattern (foundation/ws_pool/ws_pool.h), which has no
// built-in reduction primitive and has to hand-roll the same
// partial-sums-then-combine shape `#pragma omp parallel for
// reduction(+:sum)` does for free.
//
// TOOLCHAIN NOTE (read before trusting timing numbers): Apple clang does
// NOT ship libomp -- `#pragma omp` compiles cleanly without -fopenmp (an
// unrecognized pragma is legal, silently-ignored C++), so the "OpenMP"
// path below degrades to a SERIAL for-loop when built without libomp
// linked, rather than failing to compile. This file detects that at
// runtime via `_OPENMP` (defined by the compiler only when a real OpenMP
// implementation is active) and labels its own output accordingly, so a
// run on this Mac (no libomp installed -- ask-before-install declined
// this session, see CLAUDE.md's Phase 22 update) is HONEST about running
// the OpenMP path serially, not silently mislabeled as parallel. This is
// itself a real, worth-knowing platform gotcha: an unguarded `#pragma
// omp` build is a silent correctness trap for anyone who assumes a
// missing -fopenmp flag would be a build error.
//
// Build with real OpenMP once libomp is installed:
//   brew install libomp   # this session confirmed via `brew info libomp`
//                          # that a real, current, bottled formula exists
//                          # (23.1.1, keg-only) -- same "toolchain exists,
//                          # just not installed this session" situation as
//                          # step 1's open-mpi.
//   cmake --preset debug && cmake --build --preset debug --target openmp_reduce
#include "ws_pool/ws_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// foundation::WorkStealingPool has no built-in reduction primitive (see
// ws_pool.h's own doc comment: parallel_for + TaskGroup are the only
// primitives), so this hand-rolls the same "each shard reduces locally,
// then combine under a lock" shape reduction() gives OpenMP for free --
// exactly the asymmetry PLAN.md step 2 asks this comparison to surface.
double ws_pool_parallel_sum(foundation::WorkStealingPool &pool,
                             const std::vector<float> &data,
                             int num_shards) {
  std::vector<double> partial(static_cast<size_t>(num_shards), 0.0);
  size_t n = data.size();
  size_t shard_size = (n + static_cast<size_t>(num_shards) - 1) /
                       static_cast<size_t>(num_shards);

  pool.parallel_for(static_cast<size_t>(num_shards), [&](size_t shard) {
    size_t begin = shard * shard_size;
    size_t end = std::min(begin + shard_size, n);
    double local = 0.0;
    for (size_t i = begin; i < end; ++i)
      local += static_cast<double>(data[i]);
    partial[shard] = local;
  });

  double total = 0.0;
  for (double p : partial) total += p;
  return total;
}

double omp_parallel_sum(const std::vector<float> &data) {
  double sum = 0.0;
  int n = static_cast<int>(data.size());
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(data[static_cast<size_t>(i)]);
  }
  return sum;
}

// Same indexed-loop shape as omp_parallel_sum (not a range-based for) so
// the two are a fair per-iteration comparison -- an earlier version used
// `for (float v : data)` here and saw a nonsensical >1x "speedup" for the
// omp-pragma path even with the pragma stripped to a serial no-op (no
// _OPENMP defined); root cause was this loop-shape mismatch dominating at
// -O0 (debug preset), not any real difference in work done. See README.
double serial_sum(const std::vector<float> &data) {
  double sum = 0.0;
  int n = static_cast<int>(data.size());
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(data[static_cast<size_t>(i)]);
  }
  return sum;
}

}  // namespace

int main() {
  constexpr size_t kN = 20'000'000;
  std::vector<float> data(kN);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : data) v = dist(rng);

  double expected = serial_sum(data);

  foundation::WorkStealingPool pool;
  int num_shards = static_cast<int>(pool.thread_count()) * 4;

  auto t0 = Clock::now();
  double ws_sum = ws_pool_parallel_sum(pool, data, num_shards);
  auto t1 = Clock::now();
  double omp_result = omp_parallel_sum(data);
  auto t2 = Clock::now();
  double serial_result = serial_sum(data);
  auto t3 = Clock::now();

  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double ws_ms = ms(t0, t1);
  double omp_ms = ms(t1, t2);
  double serial_ms = ms(t2, t3);

  bool ws_ok = std::fabs(ws_sum - expected) < 1e-3 * std::fabs(expected);
  bool omp_ok = std::fabs(omp_result - expected) < 1e-3 * std::fabs(expected);
  bool serial_ok =
      std::fabs(serial_result - expected) < 1e-6 * std::fabs(expected);
  (void)serial_ok;  // serial_result == expected by construction

#ifdef _OPENMP
  bool real_openmp = true;
#else
  bool real_openmp = false;
#endif

  std::printf("openmp_reduce: N=%zu, threads=%zu, shards=%d\n", kN,
              pool.thread_count(), num_shards);
  std::printf("  serial baseline              : sum=%.6f  %.3f ms\n",
              serial_result, serial_ms);
  std::printf(
      "  foundation::WorkStealingPool : sum=%.6f  %s  %.3f ms  (%.2fx vs "
      "serial)\n",
      ws_sum, ws_ok ? "PASS" : "FAIL", ws_ms, serial_ms / ws_ms);
  std::printf(
      "  #pragma omp parallel for     : sum=%.6f  %s  %.3f ms  (%.2fx vs "
      "serial)  [_OPENMP %s -> %s]\n",
      omp_result, omp_ok ? "PASS" : "FAIL", omp_ms, serial_ms / omp_ms,
      real_openmp ? "defined" : "NOT defined",
      real_openmp ? "real parallel OpenMP" : "SERIAL FALLBACK (no libomp linked)");

  if (!real_openmp) {
    std::printf(
        "  NOTE: this build has no libomp -- the omp pragma above compiled "
        "as a no-op and ran serially. Its timing is NOT a real OpenMP "
        "speedup measurement; see README for the toolchain gate.\n");
  }

  return (ws_ok && omp_ok) ? 0 : 1;
}
