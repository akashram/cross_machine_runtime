// raft_test.cpp — 3-node cluster over a real loopback TCP mesh: (1) wait
// for exactly one leader to be elected, (2) propose a command on the
// leader and verify it commits identically, in order, on every node's
// onCommit callback, (3) stop the leader, verify a new leader is elected
// among the survivors and can still commit.

#include "raft.h"

#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

int findLeader(std::vector<std::unique_ptr<raft::RaftNode>> &nodes, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    int leaderCount = 0, leaderIdx = -1;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i] && nodes[i]->state() == raft::RaftState::LEADER) { ++leaderCount; leaderIdx = static_cast<int>(i); }
    }
    if (leaderCount == 1) return leaderIdx;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return -1;
}

} // namespace

int main() {
  constexpr int kWorldSize = 3;
  constexpr uint16_t kBasePort = 35601;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::unique_ptr<raft::RaftNode>> nodes;
  for (int r = 0; r < kWorldSize; ++r)
    nodes.push_back(std::make_unique<raft::RaftNode>(*channels[r]));

  std::mutex logMu;
  std::vector<std::vector<std::string>> committedPerNode(kWorldSize);
  for (int r = 0; r < kWorldSize; ++r) {
    nodes[static_cast<size_t>(r)]->onCommit([&, r](const raft::LogEntry &e) {
      std::lock_guard<std::mutex> lock(logMu);
      committedPerNode[static_cast<size_t>(r)].push_back(e.command);
    });
  }
  for (auto &n : nodes) n->start();

  int failures = 0;
  auto expect = [&](const char *name, bool cond) {
    std::printf("%-55s %s\n", name, cond ? "OK" : "FAIL");
    if (!cond) ++failures;
  };

  int leader = findLeader(nodes, 2000);
  expect("exactly one leader elected within 2s", leader >= 0);
  if (leader < 0) { std::printf("FAIL\n"); return 1; }
  std::printf("  (leader is node %d, term %llu)\n", leader,
              static_cast<unsigned long long>(nodes[static_cast<size_t>(leader)]->currentTerm()));

  bool committed = nodes[static_cast<size_t>(leader)]->propose("set x=1", 2000);
  expect("propose('set x=1') commits", committed);
  std::this_thread::sleep_for(std::chrono::milliseconds(200)); // let commit propagate to all nodes' onCommit

  {
    std::lock_guard<std::mutex> lock(logMu);
    for (int r = 0; r < kWorldSize; ++r)
      expect(("node " + std::to_string(r) + " committed exactly [\"set x=1\"]").c_str(),
             committedPerNode[static_cast<size_t>(r)] == std::vector<std::string>{"set x=1"});
  }

  // Failover: stop the leader, expect a new leader among the survivors,
  // and expect that new leader can still commit.
  int oldLeader = leader;
  nodes[static_cast<size_t>(oldLeader)]->stop();

  std::vector<raft::RaftNode *> survivors;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
  int newLeader = -1;
  while (std::chrono::steady_clock::now() < deadline && newLeader < 0) {
    for (int r = 0; r < kWorldSize; ++r) {
      if (r == oldLeader) continue;
      if (nodes[static_cast<size_t>(r)]->state() == raft::RaftState::LEADER) { newLeader = r; break; }
    }
    if (newLeader < 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  expect("new leader elected among survivors after leader stop", newLeader >= 0 && newLeader != oldLeader);

  if (newLeader >= 0) {
    bool committed2 = nodes[static_cast<size_t>(newLeader)]->propose("set y=2", 2000);
    expect("new leader commits 'set y=2'", committed2);
  }

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");

  for (int r = 0; r < kWorldSize; ++r) if (r != oldLeader) nodes[static_cast<size_t>(r)]->stop();
  return failures == 0 ? 0 : 1;
}
