// halving_doubling_test.cpp — correctness check over a real loopback TCP
// mesh, same shape as ring_allreduce_test.cpp. See README.md for output.

#include "halving_doubling.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <vector>

int main() {
  constexpr int kWorldSize = 4; // must be a power of 2
  constexpr size_t kCount = 1 << 20; // divisible by kWorldSize
  constexpr uint16_t kBasePort = 35101;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);

  std::vector<std::future<double>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r]() -> double {
      std::vector<float> buf(kCount);
      for (size_t i = 0; i < kCount; ++i) buf[i] = static_cast<float>(r + 1);

      auto start = std::chrono::steady_clock::now();
      halving_doubling_allreduce(buf.data(), kCount, *ch);
      double elapsedSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

      float expected = static_cast<float>(kWorldSize * (kWorldSize + 1) / 2);
      for (size_t i = 0; i < kCount; ++i) {
        if (std::abs(buf[i] - expected) > 1e-3f) {
          std::printf("rank %d: MISMATCH at %zu: got %f want %f\n", r, i, buf[i], expected);
          return -1.0;
        }
      }
      double gb = (2.0 * (kWorldSize - 1) / kWorldSize) * kCount * sizeof(float) / 1e9;
      return gb / elapsedSec;
    }));
  }

  bool allOk = true;
  for (int r = 0; r < kWorldSize; ++r) {
    double bw = results[r].get();
    if (bw < 0) { allOk = false; continue; }
    std::printf("rank %d: correct, effective bandwidth %.3f GB/s\n", r, bw);
  }
  std::printf("%s\n", allOk ? "PASS" : "FAIL");
  return allOk ? 0 : 1;
}
