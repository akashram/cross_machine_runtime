//===- chandy_lamport.h - Distributed snapshot without stopping execution -===//
//
// The classic algorithm (Chandy & Lamport, 1985), implemented over
// networking/common::Channel with the assumption Channel's TCP transport
// actually satisfies (FIFO, reliable per-pair delivery — the algorithm's
// only real precondition). Any node can initiate; a snapshot recorded
// this way is guaranteed *consistent* (there's a global cut it
// corresponds to) without ever pausing the application — the whole point
// versus a naive "stop the world, read every process's state" snapshot.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "channel.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace snapshot {

struct SnapshotResult {
  int64_t localState = 0;
  // Per incoming peer, the application messages that were "in flight" on
  // that channel at the moment of the consistent cut this snapshot
  // captures — recorded because they arrived after this process started
  // recording but before that peer's marker arrived.
  std::unordered_map<int, std::vector<int64_t>> channelMessages;
};

// One int64 "amount" per application message — deliberately narrow (this
// is the classic bank-account-transfer example the algorithm is usually
// taught with) rather than a generic byte-payload channel, so the
// invariant this step's test checks (sum of recorded state == sum of
// initial state, regardless of which messages were in flight when the
// snapshot ran) is exactly one arithmetic check, not a generic
// serialization comparison.
class ChandyLamportNode {
public:
  using GetLocalState = std::function<int64_t()>;
  using OnTransfer = std::function<void(int64_t amount)>; // applies to local state

  ChandyLamportNode(netcommon::Channel &channel, GetLocalState getLocalState,
                     OnTransfer onTransfer);
  ~ChandyLamportNode();

  void start(); // spawns one receiver thread per peer
  void stop();

  void sendTransfer(int peer, int64_t amount);

  // Runs `action` (expected to debit local state and call sendTransfer)
  // atomically with respect to this node's snapshot state machine.
  // Required for correctness, not just thread-safety: the algorithm's
  // FIFO precondition ("a message this process sends after taking its
  // local snapshot arrives after the corresponding marker") only holds if
  // "record state + broadcast markers" can't be interleaved with a
  // concurrent application send — see the comment in
  // chandy_lamport.cpp's initiateSnapshot(). Any caller mixing
  // sendTransfer with concurrent snapshots MUST route sends through here,
  // not call sendTransfer directly.
  template <typename Fn>
  void atomically(Fn &&action) {
    std::lock_guard<std::mutex> lock(mu_);
    action();
  }

  // Initiates a snapshot from this node and blocks until it completes
  // (every neighbor's marker has come back around). Only one snapshot
  // may be in flight at a time per process — see chandy_lamport.cpp.
  SnapshotResult initiateSnapshot();

  // For a node that didn't initiate: blocks until a snapshot triggered by
  // some other node (its first marker arriving here) completes locally.
  SnapshotResult waitForPassiveSnapshot();

private:
  void receiverLoop(int peer);
  void handleMarker(int fromPeer);
  void handleTransfer(int fromPeer, int64_t amount);

  netcommon::Channel &channel_;
  GetLocalState getLocalState_;
  OnTransfer onTransfer_;
  std::vector<int> peers_;
  std::vector<std::thread> receiverThreads_;
  std::atomic<bool> running_{false};

  std::mutex mu_;
  std::condition_variable snapshotDone_;
  bool snapshotting_ = false;
  SnapshotResult current_;
  std::vector<bool> markerSeen_; // indexed by peer rank
  int markersOutstanding_ = 0;
  bool resultReady_ = false;
};

} // namespace snapshot
