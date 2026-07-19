//===- chandy_lamport.cpp - Step 18 implementation ------------------------===//
//
// One receiver thread per peer (Channel's recv is blocking and per-peer,
// so this is the straightforward way to react to whichever peer sends
// next without polling); all marker/transfer handling funnels through
// handleMarker/handleTransfer under `mu_`, so the actual snapshot state
// machine is single-threaded in effect even though messages arrive
// concurrently from multiple receiver threads.
//
//===----------------------------------------------------------------------===//

#include "chandy_lamport.h"

#include <arpa/inet.h>

#include <cstring>

namespace snapshot {namespace {

constexpr uint8_t kTagMarker = 0;
constexpr uint8_t kTagTransfer = 1;
constexpr uint8_t kTagShutdown = 2;

uint64_t hton64(uint64_t v) {
  uint32_t hi = htonl(static_cast<uint32_t>(v >> 32));
  uint32_t lo = htonl(static_cast<uint32_t>(v & 0xffffffffu));
  return (static_cast<uint64_t>(lo) << 32) | hi; // swap halves after each is byte-swapped
}
uint64_t ntoh64(uint64_t v) { return hton64(v); } // symmetric

struct WireMsg {
  uint8_t tag;
  int64_t amount;
};

void sendMsg(netcommon::Channel &ch, int peer, uint8_t tag, int64_t amount) {
  uint8_t buf[9];
  buf[0] = tag;
  uint64_t net = hton64(static_cast<uint64_t>(amount));
  std::memcpy(buf + 1, &net, 8);
  ch.send(peer, buf, sizeof(buf));
}

WireMsg recvMsg(netcommon::Channel &ch, int peer) {
  uint8_t buf[9];
  ch.recv(peer, buf, sizeof(buf));
  uint64_t net;
  std::memcpy(&net, buf + 1, 8);
  return {buf[0], static_cast<int64_t>(ntoh64(net))};
}

} // namespace

ChandyLamportNode::ChandyLamportNode(netcommon::Channel &channel, GetLocalState getLocalState,
                                       OnTransfer onTransfer)
    : channel_(channel), getLocalState_(std::move(getLocalState)), onTransfer_(std::move(onTransfer)),
      markerSeen_(static_cast<size_t>(channel.world_size()), false) {
  for (int p = 0; p < channel_.world_size(); ++p)
    if (p != channel_.rank()) peers_.push_back(p);
}

ChandyLamportNode::~ChandyLamportNode() { stop(); }

void ChandyLamportNode::start() {
  running_ = true;
  for (int peer : peers_) receiverThreads_.emplace_back(&ChandyLamportNode::receiverLoop, this, peer);
}

void ChandyLamportNode::stop() {
  if (!running_.exchange(false)) return; // already stopped
  // Channel::recv gives no cancellation mechanism — a receiver thread
  // blocked waiting on `peer` only unblocks when `peer` sends something.
  // So shutdown is itself a protocol: tell every peer we're done, then
  // wait for our receiver threads to see the *peer's* shutdown message
  // in return. Callers must invoke stop() on every node concurrently
  // (see chandy_lamport_test.cpp) — sequential stop() calls would
  // deadlock, each waiting on a peer that hasn't been told to stop yet.
  for (int p : peers_) sendMsg(channel_, p, kTagShutdown, 0);
  for (auto &t : receiverThreads_) if (t.joinable()) t.join();
  receiverThreads_.clear();
}

void ChandyLamportNode::receiverLoop(int peer) {
  for (;;) {
    WireMsg msg;
    try {
      msg = recvMsg(channel_, peer);
    } catch (...) {
      return; // channel closed — normal at final teardown
    }
    if (msg.tag == kTagShutdown) return;
    if (msg.tag == kTagMarker) handleMarker(peer);
    else handleTransfer(peer, msg.amount);
  }
}

void ChandyLamportNode::sendTransfer(int peer, int64_t amount) {
  sendMsg(channel_, peer, kTagTransfer, amount);
}

void ChandyLamportNode::handleTransfer(int fromPeer, int64_t amount) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    // Recording this channel means: we've started our own snapshot
    // (snapshotting_) and haven't yet seen fromPeer's marker.
    if (snapshotting_ && !markerSeen_[static_cast<size_t>(fromPeer)])
      current_.channelMessages[fromPeer].push_back(amount);
  }
  onTransfer_(amount); // always applied to local state — this is normal operation, snapshot or not
}

void ChandyLamportNode::handleMarker(int fromPeer) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!snapshotting_) {
    // First marker seen anywhere: this is how a passive (non-initiating)
    // node's snapshot begins.
    snapshotting_ = true;
    current_ = SnapshotResult{};
    current_.localState = getLocalState_(); // record local state NOW, before processing anything further
    std::fill(markerSeen_.begin(), markerSeen_.end(), false);
    markersOutstanding_ = static_cast<int>(peers_.size()) - 1; // fromPeer's marker already accounted for below
    markerSeen_[static_cast<size_t>(fromPeer)] = true;
    // Propagate to every neighbor — deliberately still holding `mu_`
    // while doing this socket I/O. That's the correctness-critical part:
    // any concurrent application send guarded by atomically() (see
    // header) is now strictly ordered either entirely before this whole
    // record-state-and-broadcast-markers sequence, or entirely after it,
    // for *every* peer uniformly. Without that, a send squeezed into the
    // gap between "record state" and "marker reaches peer X" could beat
    // the marker to peer X while this process's recorded local state
    // already reflects having sent it — double-counting that message in
    // both the recorded state and, at the receiver, before its marker.
    for (int p : peers_) sendMsg(channel_, p, kTagMarker, 0);
    if (markersOutstanding_ == 0) { resultReady_ = true; snapshotDone_.notify_all(); }
  } else if (!markerSeen_[static_cast<size_t>(fromPeer)]) {
    // Second (or later, from a *different* peer) marker: finalize that
    // channel's recording — no more messages from fromPeer belong to
    // this snapshot's consistent cut.
    markerSeen_[static_cast<size_t>(fromPeer)] = true;
    if (--markersOutstanding_ == 0) { resultReady_ = true; snapshotDone_.notify_all(); }
  }
  // A duplicate marker from an already-seen peer can't happen with a
  // reliable FIFO channel and one initiateSnapshot() in flight at a time.
}

SnapshotResult ChandyLamportNode::initiateSnapshot() {
  std::unique_lock<std::mutex> lock(mu_);
  snapshotting_ = true;
  current_ = SnapshotResult{};
  current_.localState = getLocalState_();
  std::fill(markerSeen_.begin(), markerSeen_.end(), false);
  markersOutstanding_ = static_cast<int>(peers_.size());
  resultReady_ = false;
  // Same correctness-critical ordering as handleMarker's first branch:
  // record state and broadcast markers as one atomic unit against
  // atomically()-guarded application sends.
  for (int p : peers_) sendMsg(channel_, p, kTagMarker, 0);

  snapshotDone_.wait(lock, [&] { return resultReady_; });
  SnapshotResult result = current_;
  snapshotting_ = false;
  return result;
}

SnapshotResult ChandyLamportNode::waitForPassiveSnapshot() {
  std::unique_lock<std::mutex> lock(mu_);
  snapshotDone_.wait(lock, [&] { return resultReady_; });
  SnapshotResult result = current_;
  snapshotting_ = false;
  resultReady_ = false;
  return result;
}

} // namespace snapshot
