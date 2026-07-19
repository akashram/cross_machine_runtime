// tree_allreduce_test.cpp — correctness check over a real loopback TCP
// mesh, including a non-power-of-2 world size (5) to exercise the bounds
// checks in both tree_reduce_to_root and tree_broadcast.

#include "tree_allreduce.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <vector>

namespace {
bool runTrial(int worldSize, uint16_t basePort) {
  constexpr size_t kCount = 1 << 16; // 64K floats

  auto channels = netcommon::make_tcp_loopback_mesh(worldSize, basePort);
  std::vector<std::future<bool>> results;
  for (int r = 0; r < worldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r, worldSize]() {
      std::vector<float> buf(kCount);
      for (size_t i = 0; i < kCount; ++i) buf[i] = static_cast<float>(r + 1);
      tree_allreduce(buf.data(), kCount, *ch);
      float expected = static_cast<float>(worldSize * (worldSize + 1) / 2);
      for (size_t i = 0; i < kCount; ++i)
        if (std::abs(buf[i] - expected) > 1e-3f) {
          std::printf("rank %d: MISMATCH at %zu: got %f want %f\n", r, i, buf[i], expected);
          return false;
        }
      return true;
    }));
  }
  bool allOk = true;
  for (auto &f : results) allOk = f.get() && allOk;
  std::printf("world_size=%d: %s\n", worldSize, allOk ? "PASS" : "FAIL");
  return allOk;
}
} // namespace

int main() {
  bool ok = true;
  ok = runTrial(4, 35201) && ok; // power of 2
  ok = runTrial(5, 35211) && ok; // non-power-of-2 — exercises bounds checks
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
