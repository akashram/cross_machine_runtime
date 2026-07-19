// collectives_test.cpp — correctness check for Broadcast, ReduceScatter,
// and AllGather over a real loopback TCP mesh.

#include "collectives.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <vector>

namespace {
constexpr int kWorldSize = 4;

bool testBroadcast(uint16_t basePort) {
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, basePort);
  constexpr size_t kCount = 1000;
  std::vector<std::future<bool>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r]() {
      std::vector<float> buf(kCount, r == 2 ? 42.0f : -1.0f); // root = rank 2
      collectives::Broadcast(buf.data(), kCount, *ch, /*root=*/2);
      for (float v : buf) if (v != 42.0f) return false;
      return true;
    }));
  }
  bool ok = true;
  for (auto &f : results) ok = f.get() && ok;
  std::printf("Broadcast: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool testReduceScatter(uint16_t basePort) {
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, basePort);
  constexpr size_t kCount = 4096; // divisible by kWorldSize
  std::vector<std::future<bool>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r]() {
      std::vector<float> buf(kCount);
      for (size_t i = 0; i < kCount; ++i) buf[i] = static_cast<float>(r + 1);
      collectives::ReduceScatter(buf.data(), kCount, *ch);
      int slot = (r + 1) % kWorldSize; // see collectives.h: owned chunk is (rank+1)%N
      size_t sliceStart = kCount / kWorldSize * static_cast<size_t>(slot);
      size_t sliceEnd = kCount / kWorldSize * static_cast<size_t>(slot + 1);
      float expected = static_cast<float>(kWorldSize * (kWorldSize + 1) / 2);
      for (size_t i = sliceStart; i < sliceEnd; ++i)
        if (std::abs(buf[i] - expected) > 1e-3f) return false;
      return true;
    }));
  }
  bool ok = true;
  for (auto &f : results) ok = f.get() && ok;
  std::printf("ReduceScatter: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool testAllGather(uint16_t basePort) {
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, basePort);
  constexpr size_t kSendCount = 256;
  std::vector<std::future<bool>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r]() {
      std::vector<float> sendBuf(kSendCount, static_cast<float>(r));
      std::vector<float> recvBuf(kSendCount * kWorldSize);
      collectives::AllGather(sendBuf.data(), kSendCount, recvBuf.data(), *ch);
      for (int src = 0; src < kWorldSize; ++src) {
        int slot = (src + 1) % kWorldSize; // see collectives.h: placed at (rank+1)%N
        for (size_t i = 0; i < kSendCount; ++i)
          if (recvBuf[static_cast<size_t>(slot) * kSendCount + i] != static_cast<float>(src))
            return false;
      }
      return true;
    }));
  }
  bool ok = true;
  for (auto &f : results) ok = f.get() && ok;
  std::printf("AllGather: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}
} // namespace

int main() {
  bool ok = true;
  ok = testBroadcast(35301) && ok;
  ok = testReduceScatter(35311) && ok;
  ok = testAllGather(35321) && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
