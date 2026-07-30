// channel_peerhosts_test.cpp — sanity test for the peer_hosts-vector
// TcpChannel constructor added for distributed_training/training_worker
// (Phase 16 gap fix): confirms the new overload wires up correctly
// (binds "0.0.0.0", dials each lower rank at peer_hosts[j]) before
// training_worker's own real multi-PROCESS run depends on it. Ranks are
// threads here (same loopback-mesh style as channel_test.cpp) --
// training_worker's own README has the real cross-PROCESS proof.
#include "channel.h"

#include <cstdio>
#include <future>
#include <vector>

int main() {
  constexpr int kWorldSize = 3;
  constexpr uint16_t kBasePort = 34600;
  std::vector<std::string> peer_hosts(kWorldSize, "127.0.0.1");

  std::vector<std::future<std::unique_ptr<netcommon::Channel>>> futures;
  for (int r = 0; r < kWorldSize; ++r) {
    futures.push_back(std::async(std::launch::async, [r, kWorldSize, kBasePort, &peer_hosts] {
      return std::unique_ptr<netcommon::Channel>(
          std::make_unique<netcommon::TcpChannel>(r, kWorldSize, kBasePort, peer_hosts));
    }));
  }
  std::vector<std::unique_ptr<netcommon::Channel>> channels;
  for (auto &f : futures) channels.push_back(f.get());
  std::printf("peer_hosts mesh established: %d ranks, each dialed by hostname\n", kWorldSize);

  std::vector<std::future<bool>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [ch, r]() {
      bool ok = true;
      for (int p = 0; p < kWorldSize; ++p) {
        if (p == r) continue;
        int32_t sendVal = r, recvVal = -1;
        if (r < p) {
          ch->send(p, &sendVal, sizeof(sendVal));
          ch->recv(p, &recvVal, sizeof(recvVal));
        } else {
          ch->recv(p, &recvVal, sizeof(recvVal));
          ch->send(p, &sendVal, sizeof(sendVal));
        }
        ok = ok && (recvVal == p);
      }
      return ok;
    }));
  }

  bool allOk = true;
  for (int r = 0; r < kWorldSize; ++r) {
    bool ok = results[static_cast<size_t>(r)].get();
    std::printf("rank %d: all-pairs exchange %s\n", r, ok ? "OK" : "FAILED");
    allOk = allOk && ok;
  }
  std::printf("%s\n", allOk ? "PASS" : "FAIL");
  return allOk ? 0 : 1;
}
