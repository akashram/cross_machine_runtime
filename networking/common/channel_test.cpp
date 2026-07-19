// channel_test.cpp — sanity test for TcpChannel: builds a 4-rank loopback
// TCP mesh (real sockets, real accept()/connect(), real localhost round
// trips) and does an all-pairs exchange, verifying every rank receives
// exactly its peer's rank id from every peer. This is what's meant by
// "build+run locally" throughout Phase 5's READMEs — see
// networking/common/README.md for captured output.

#include "channel.h"

#include <cstdio>
#include <future>
#include <vector>

int main() {
  constexpr int kWorldSize = 4;
  constexpr uint16_t kBasePort = 34567;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::printf("mesh established: %d ranks over real TCP sockets (127.0.0.1:%d-%d)\n",
              kWorldSize, kBasePort, kBasePort + kWorldSize - 1);

  std::vector<std::future<bool>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[r].get();
    results.push_back(std::async(std::launch::async, [ch, r]() {
      bool ok = true;
      for (int p = 0; p < kWorldSize; ++p) {
        if (p == r) continue;
        int32_t sendVal = r, recvVal = -1;
        // Deadlock-safe ordering: lower rank sends first, higher rank
        // receives first — the same pattern every collective in this
        // directory tree uses (see channel.h's deadlock note).
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
    bool ok = results[r].get();
    std::printf("rank %d: all-pairs exchange %s\n", r, ok ? "OK" : "FAILED");
    allOk = allOk && ok;
  }

  std::printf("%s\n", allOk ? "PASS" : "FAIL");
  return allOk ? 0 : 1;
}
