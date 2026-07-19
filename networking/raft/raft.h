//===- raft.h - Raft consensus (leader election + log replication) ------===//
//
// Built on networking/common::Channel, like every other portable Phase 5
// step — the stub's original `RaftNode(id, peer_addrs)` constructor
// (address strings) is replaced with `RaftNode(Channel&)`, matching
// chandy_lamport's and every collective's pattern: the transport is
// injected once, `channel.rank()`/`world_size()` give identity and peer
// count. Scope: leader election + log replication, validated against the
// TLA+ spec (tla_raft/, step 20). Cluster membership changes (PLAN.md
// mentions them as part of "Raft from scratch") are NOT implemented —
// documented gap, same "mechanism first, revisit with a real need" stance
// as every other documented simplification in this project. See
// networking/DESIGN.md.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "channel.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace raft {

enum class RaftState { FOLLOWER, CANDIDATE, LEADER };

struct LogEntry {
  uint64_t term = 0;
  uint64_t index = 0;
  std::string command;
};

class RaftNode {
public:
  using CommitCallback = std::function<void(const LogEntry &)>;

  explicit RaftNode(netcommon::Channel &channel);
  ~RaftNode();

  void start();
  void stop();

  // Proposes a command on the leader. Blocks until the resulting log
  // entry commits (a majority has replicated it) or `timeout_ms` elapses.
  // Returns false immediately (no wait) if this node isn't currently the
  // leader — callers are expected to retry against whichever node
  // `leaderId()` reports, same as any real Raft client.
  bool propose(const std::string &command, uint64_t timeout_ms = 5000);

  RaftState state() const;
  int leaderId() const;
  uint64_t currentTerm() const;

  void onCommit(CommitCallback cb);

private:
  struct PeerRpcState {
    uint64_t nextIndex = 1;
    uint64_t matchIndex = 0;
  };

  void receiverLoop(int peer);
  void tickerLoop();
  void becomeFollower(uint64_t newTerm);
  void becomeCandidate();
  void becomeLeader();
  void startElection();
  void sendHeartbeats();
  void sendAppendEntriesTo(int peer);
  void handleRequestVote(int fromPeer, uint64_t term, int candidateId, uint64_t lastLogIndex,
                          uint64_t lastLogTerm);
  void handleRequestVoteResponse(int fromPeer, uint64_t term, bool voteGranted);
  void handleAppendEntries(int fromPeer, uint64_t term, int leaderId, uint64_t prevLogIndex,
                            uint64_t prevLogTerm, const std::vector<LogEntry> &entries,
                            uint64_t leaderCommit);
  void handleAppendEntriesResponse(int fromPeer, uint64_t term, bool success, uint64_t matchIndex);
  void advanceCommitIndex(); // caller holds mu_; newly-committed entries are drained by the caller
  int64_t randomElectionTimeoutMs() const;

  netcommon::Channel &channel_;
  int selfId_;
  std::vector<int> peers_;
  std::atomic<bool> running_{false};
  std::vector<std::thread> receiverThreads_;
  std::thread tickerThread_;
  std::vector<std::mutex> sendMu_; // one per peer, guards interleaved writes to its socket

  mutable std::mutex mu_;
  std::condition_variable commitCv_;
  RaftState state_ = RaftState::FOLLOWER;
  uint64_t currentTerm_ = 0;
  int votedFor_ = -1;
  int leaderId_ = -1;
  std::vector<LogEntry> log_; // 1-indexed conceptually; log_[0] is a term-0 sentinel at index 0
  uint64_t commitIndex_ = 0;
  uint64_t lastApplied_ = 0;
  std::vector<PeerRpcState> peerState_; // indexed by peer rank
  int votesGranted_ = 0;
  std::chrono::steady_clock::time_point electionDeadline_;
  CommitCallback onCommit_;
};

} // namespace raft
