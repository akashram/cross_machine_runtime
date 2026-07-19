// chandy_lamport_test.cpp — the classic bank-account invariant test:
// 3 ranks each start with balance=100 (total=300), background threads on
// every rank send small transfers to random peers *concurrently* with
// rank 0 initiating a snapshot. Regardless of which transfers happen to
// be pre-snapshot, post-snapshot, or genuinely in-flight across the cut,
// sum(every rank's recorded local state) + sum(every recorded in-flight
// channel message) must equal the original total — that's the actual
// guarantee Chandy-Lamport makes, and the only thing this test needs to
// check (not any specific interleaving outcome).

#include "chandy_lamport.h"

#include <atomic>
#include <cstdio>
#include <future>
#include <random>
#include <thread>
#include <vector>

int main() {
  constexpr int kWorldSize = 3;
  constexpr int64_t kInitialBalance = 100;
  constexpr uint16_t kBasePort = 35501;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::atomic<int64_t>> balances(kWorldSize);
  for (auto &b : balances) b = kInitialBalance;

  std::vector<std::unique_ptr<snapshot::ChandyLamportNode>> nodes;
  for (int r = 0; r < kWorldSize; ++r) {
    nodes.push_back(std::make_unique<snapshot::ChandyLamportNode>(
        *channels[r],
        [&balances, r] { return balances[static_cast<size_t>(r)].load(); },
        [&balances, r](int64_t amount) { balances[static_cast<size_t>(r)].fetch_add(amount); }));
  }
  for (auto &n : nodes) n->start();

  // Background load: every rank sends small transfers to random peers
  // throughout, overlapping with the snapshot below.
  std::atomic<bool> loadRunning{true};
  std::vector<std::thread> loadThreads;
  for (int r = 0; r < kWorldSize; ++r) {
    loadThreads.emplace_back([&, r] {
      std::mt19937 rng(static_cast<unsigned>(r) * 7919u + 1);
      std::uniform_int_distribution<int> peerDist(0, kWorldSize - 2);
      for (int i = 0; i < 300 && loadRunning.load(); ++i) {
        int peer = peerDist(rng);
        if (peer >= r) peer++; // exclude self
        nodes[static_cast<size_t>(r)]->atomically([&] {
          if (balances[static_cast<size_t>(r)].load() >= 1) {
            balances[static_cast<size_t>(r)].fetch_sub(1);
            nodes[static_cast<size_t>(r)]->sendTransfer(peer, 1);
          }
        });
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
  }

  // Rank 0 initiates; ranks 1 and 2 wait for the passive snapshot —
  // launched concurrently, matching how this actually gets used (nobody
  // knows in advance when a remote-initiated snapshot's marker will
  // arrive).
  auto initiatorFuture = std::async(std::launch::async, [&] { return nodes[0]->initiateSnapshot(); });
  auto passive1Future = std::async(std::launch::async, [&] { return nodes[1]->waitForPassiveSnapshot(); });
  auto passive2Future = std::async(std::launch::async, [&] { return nodes[2]->waitForPassiveSnapshot(); });

  snapshot::SnapshotResult results[3] = {initiatorFuture.get(), passive1Future.get(), passive2Future.get()};

  loadRunning = false;
  for (auto &t : loadThreads) t.join();

  int64_t total = 0;
  for (int r = 0; r < kWorldSize; ++r) {
    total += results[r].localState;
    std::printf("rank %d: recorded local state = %lld\n", r, static_cast<long long>(results[r].localState));
    for (auto &[peer, amounts] : results[r].channelMessages) {
      int64_t channelSum = 0;
      for (int64_t a : amounts) channelSum += a;
      total += channelSum;
      if (!amounts.empty())
        std::printf("rank %d: channel from %d recorded %zu in-flight message(s), sum=%lld\n", r, peer,
                    amounts.size(), static_cast<long long>(channelSum));
    }
  }

  std::printf("total recorded = %lld (expected %lld)\n", static_cast<long long>(total),
              static_cast<long long>(kWorldSize * kInitialBalance));
  bool ok = total == kWorldSize * kInitialBalance;
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::vector<std::future<void>> stopFutures;
  for (auto &n : nodes) stopFutures.push_back(std::async(std::launch::async, [&n] { n->stop(); }));
  for (auto &f : stopFutures) f.get();

  return ok ? 0 : 1;
}
