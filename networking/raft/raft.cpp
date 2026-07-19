//===- raft.cpp - Step 19 implementation ----------------------------------===//
//
// One receiver thread per peer (same shape as chandy_lamport.cpp) feeds a
// single mutex-guarded state machine — every RequestVote/AppendEntries
// RPC and its response is just a tagged, length-prefixed frame on that
// peer's Channel socket; there's no separate request/response
// correlation because a node only ever has one outstanding logical
// request per peer at a time (heartbeats/log pushes are level-triggered
// by the ticker, not queued).
//
// Safety-critical detail worth calling out: `advanceCommitIndex` only
// commits an index whose entry's *term equals currentTerm_* — the classic
// Raft rule (§5.4.2 of the paper) that a leader must never commit an
// entry from a previous term purely by counting replicas; it can only be
// committed as a side effect of committing a later entry from the
// leader's own term. Skipping this check is the single most common Raft
// implementation bug (it's *almost* always safe, which is what makes it
// dangerous) — see raft/README.md and tla_raft/Raft.tla for how this gets
// checked for real.
//
//===----------------------------------------------------------------------===//

#include "raft.h"

#include <arpa/inet.h>

#include <algorithm>
#include <cstring>
#include <random>

namespace raft {
namespace {

constexpr uint8_t kMsgShutdown = 0;
constexpr uint8_t kMsgRequestVote = 1;
constexpr uint8_t kMsgRequestVoteResp = 2;
constexpr uint8_t kMsgAppendEntries = 3;
constexpr uint8_t kMsgAppendEntriesResp = 4;

using Clock = std::chrono::steady_clock;

// --- wire format helpers: length-prefixed frames, big-endian fields ---

struct Writer {
  std::vector<uint8_t> buf;
  void u8(uint8_t v) { buf.push_back(v); }
  void u32(uint32_t v) {
    uint32_t n = htonl(v);
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&n);
    buf.insert(buf.end(), p, p + 4);
  }
  void u64(uint64_t v) {
    u32(static_cast<uint32_t>(v >> 32));
    u32(static_cast<uint32_t>(v & 0xffffffffu));
  }
  void str(const std::string &s) {
    u32(static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
  }
};

struct Reader {
  const uint8_t *p;
  size_t remaining;
  uint8_t u8() { uint8_t v = *p; p += 1; remaining -= 1; return v; }
  uint32_t u32() {
    uint32_t n; std::memcpy(&n, p, 4); p += 4; remaining -= 4;
    return ntohl(n);
  }
  uint64_t u64() {
    uint64_t hi = u32(), lo = u32();
    return (hi << 32) | lo;
  }
  std::string str() {
    uint32_t len = u32();
    std::string s(reinterpret_cast<const char *>(p), len);
    p += len; remaining -= len;
    return s;
  }
};

void sendFrame(netcommon::Channel &ch, std::mutex &sendMu, int peer, const std::vector<uint8_t> &payload) {
  std::lock_guard<std::mutex> lock(sendMu);
  uint32_t lenNet = htonl(static_cast<uint32_t>(payload.size()));
  ch.send(peer, &lenNet, 4);
  if (!payload.empty()) ch.send(peer, payload.data(), payload.size());
}

std::vector<uint8_t> recvFrame(netcommon::Channel &ch, int peer) {
  uint32_t lenNet = 0;
  ch.recv(peer, &lenNet, 4);
  uint32_t len = ntohl(lenNet);
  std::vector<uint8_t> buf(len);
  if (len > 0) ch.recv(peer, buf.data(), len);
  return buf;
}

} // namespace

RaftNode::RaftNode(netcommon::Channel &channel)
    : channel_(channel), selfId_(channel.rank()), sendMu_(static_cast<size_t>(channel.world_size())),
      peerState_(static_cast<size_t>(channel.world_size())) {
  for (int p = 0; p < channel_.world_size(); ++p)
    if (p != selfId_) peers_.push_back(p);
  log_.push_back(LogEntry{0, 0, ""}); // sentinel at index 0
  electionDeadline_ = Clock::now() + std::chrono::milliseconds(randomElectionTimeoutMs());
}

RaftNode::~RaftNode() { stop(); }

int64_t RaftNode::randomElectionTimeoutMs() const {
  thread_local std::mt19937 rng(static_cast<unsigned>(selfId_) * 2654435761u + 12345u);
  std::uniform_int_distribution<int64_t> dist(150, 300); // classic Raft paper range
  return dist(rng);
}

void RaftNode::start() {
  running_ = true;
  for (int p : peers_) receiverThreads_.emplace_back(&RaftNode::receiverLoop, this, p);
  tickerThread_ = std::thread(&RaftNode::tickerLoop, this);
}

void RaftNode::stop() {
  if (!running_.exchange(false)) return;
  // Deliberately detach, not join, the receiver threads: each is blocked
  // in Channel::recv(peer, ...), which only unblocks when `peer` sends
  // something. Joining would require every *other* node to also be
  // stopping right now and send a shutdown frame back — fine for a
  // clean-shutdown-of-the-whole-cluster test, but wrong for simulating a
  // single node's crash (raft_test.cpp's failover scenario), where the
  // survivors keep running and never send one. `channel_` outlives these
  // detached threads in every caller of this class (the test/harness
  // owns the Channel for the whole process lifetime), so a thread left
  // blocked in recv() forever is inert, not a dangling reference. The
  // ticker thread is always safe to join — it only waits on a sleep and
  // its own `running_` check, no peer cooperation required.
  for (auto &t : receiverThreads_) if (t.joinable()) t.detach();
  receiverThreads_.clear();
  if (tickerThread_.joinable()) tickerThread_.join();
}

void RaftNode::tickerLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    bool isLeader;
    bool timedOut;
    {
      std::lock_guard<std::mutex> lock(mu_);
      isLeader = (state_ == RaftState::LEADER);
      timedOut = !isLeader && Clock::now() >= electionDeadline_;
    }
    if (isLeader) sendHeartbeats();
    else if (timedOut) startElection();
  }
}

// --- role transitions (caller must hold mu_) ---

void RaftNode::becomeFollower(uint64_t newTerm) {
  state_ = RaftState::FOLLOWER;
  currentTerm_ = newTerm;
  votedFor_ = -1;
  electionDeadline_ = Clock::now() + std::chrono::milliseconds(randomElectionTimeoutMs());
}

void RaftNode::becomeCandidate() {
  state_ = RaftState::CANDIDATE;
  currentTerm_ += 1;
  votedFor_ = selfId_;
  votesGranted_ = 1; // vote for self
  leaderId_ = -1;
  electionDeadline_ = Clock::now() + std::chrono::milliseconds(randomElectionTimeoutMs());
}

void RaftNode::becomeLeader() {
  state_ = RaftState::LEADER;
  leaderId_ = selfId_;
  uint64_t nextIdx = log_.back().index + 1;
  for (auto &ps : peerState_) { ps.nextIndex = nextIdx; ps.matchIndex = 0; }
}

void RaftNode::startElection() {
  uint64_t term, lastLogIndex, lastLogTerm;
  {
    std::lock_guard<std::mutex> lock(mu_);
    becomeCandidate();
    term = currentTerm_;
    lastLogIndex = log_.back().index;
    lastLogTerm = log_.back().term;
    if (peers_.empty()) { becomeLeader(); return; } // single-node cluster: trivial majority
  }
  for (int peer : peers_) {
    Writer w;
    w.u8(kMsgRequestVote);
    w.u64(term);
    w.u32(static_cast<uint32_t>(selfId_));
    w.u64(lastLogIndex);
    w.u64(lastLogTerm);
    sendFrame(channel_, sendMu_[static_cast<size_t>(peer)], peer, w.buf);
  }
}

void RaftNode::sendHeartbeats() {
  for (int peer : peers_) sendAppendEntriesTo(peer);
}

void RaftNode::sendAppendEntriesTo(int peer) {
  uint64_t term, prevLogIndex, prevLogTerm, leaderCommit;
  std::vector<LogEntry> entries;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != RaftState::LEADER) return;
    term = currentTerm_;
    PeerRpcState &ps = peerState_[static_cast<size_t>(peer)];
    prevLogIndex = ps.nextIndex - 1;
    prevLogTerm = log_[prevLogIndex].term;
    for (uint64_t i = ps.nextIndex; i <= log_.back().index; ++i) entries.push_back(log_[i]);
    leaderCommit = commitIndex_;
  }
  Writer w;
  w.u8(kMsgAppendEntries);
  w.u64(term);
  w.u32(static_cast<uint32_t>(selfId_));
  w.u64(prevLogIndex);
  w.u64(prevLogTerm);
  w.u64(leaderCommit);
  w.u32(static_cast<uint32_t>(entries.size()));
  for (const LogEntry &e : entries) { w.u64(e.term); w.str(e.command); }
  sendFrame(channel_, sendMu_[static_cast<size_t>(peer)], peer, w.buf);
}

void RaftNode::handleRequestVote(int fromPeer, uint64_t term, int candidateId, uint64_t lastLogIndex,
                                  uint64_t lastLogTerm) {
  bool voteGranted = false;
  uint64_t responseTerm;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (term > currentTerm_) becomeFollower(term);
    if (term == currentTerm_) {
      uint64_t myLastIndex = log_.back().index, myLastTerm = log_.back().term;
      bool logOk = (lastLogTerm > myLastTerm) || (lastLogTerm == myLastTerm && lastLogIndex >= myLastIndex);
      if ((votedFor_ == -1 || votedFor_ == candidateId) && logOk) {
        votedFor_ = candidateId;
        voteGranted = true;
        electionDeadline_ = Clock::now() + std::chrono::milliseconds(randomElectionTimeoutMs());
      }
    }
    responseTerm = currentTerm_;
  }
  Writer w;
  w.u8(kMsgRequestVoteResp);
  w.u64(responseTerm);
  w.u8(voteGranted ? 1 : 0);
  sendFrame(channel_, sendMu_[static_cast<size_t>(fromPeer)], fromPeer, w.buf);
}

void RaftNode::handleRequestVoteResponse(int fromPeer, uint64_t term, bool voteGranted) {
  std::lock_guard<std::mutex> lock(mu_);
  if (term > currentTerm_) { becomeFollower(term); return; }
  if (state_ != RaftState::CANDIDATE || term != currentTerm_ || !voteGranted) return;
  (void)fromPeer;
  votesGranted_ += 1;
  if (votesGranted_ * 2 > static_cast<int>(peers_.size()) + 1) becomeLeader();
}

void RaftNode::handleAppendEntries(int fromPeer, uint64_t term, int leaderIdIncoming,
                                    uint64_t prevLogIndex, uint64_t prevLogTerm,
                                    const std::vector<LogEntry> &entries, uint64_t leaderCommit) {
  bool success = false;
  uint64_t matchIndexResult = 0, responseTerm;
  std::vector<LogEntry> newlyCommitted;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (term < currentTerm_) {
      responseTerm = currentTerm_;
    } else {
      if (term > currentTerm_ || state_ != RaftState::FOLLOWER) becomeFollower(term);
      leaderId_ = leaderIdIncoming;
      electionDeadline_ = Clock::now() + std::chrono::milliseconds(randomElectionTimeoutMs());

      if (prevLogIndex < log_.size() && log_[prevLogIndex].term == prevLogTerm) {
        size_t idx = prevLogIndex + 1;
        for (const LogEntry &e : entries) {
          if (idx < log_.size()) {
            if (log_[idx].term != e.term) { log_.resize(idx); log_.push_back(e); }
          } else {
            log_.push_back(e);
          }
          ++idx;
        }
        success = true;
        matchIndexResult = prevLogIndex + entries.size();
        if (leaderCommit > commitIndex_) commitIndex_ = std::min(leaderCommit, log_.back().index);
      }
      responseTerm = currentTerm_;

      while (lastApplied_ < commitIndex_) {
        ++lastApplied_;
        newlyCommitted.push_back(log_[lastApplied_]);
      }
    }
  }
  for (const LogEntry &e : newlyCommitted) { if (onCommit_) onCommit_(e); }
  if (!newlyCommitted.empty()) commitCv_.notify_all();

  Writer w;
  w.u8(kMsgAppendEntriesResp);
  w.u64(responseTerm);
  w.u8(success ? 1 : 0);
  w.u64(matchIndexResult);
  sendFrame(channel_, sendMu_[static_cast<size_t>(fromPeer)], fromPeer, w.buf);
}

void RaftNode::advanceCommitIndex() {
  // Caller holds mu_. Only ever commit an entry from currentTerm_
  // directly (Raft §5.4.2) — see file-level comment.
  for (uint64_t n = log_.back().index; n > commitIndex_; --n) {
    if (log_[n].term != currentTerm_) continue;
    int count = 1; // leader itself
    for (const auto &ps : peerState_) if (ps.matchIndex >= n) ++count;
    // peerState_ has an unused slot for selfId_ too (sized world_size);
    // that slot's matchIndex stays 0 and never reaches a positive n, so
    // it never double-counts the leader.
    if (count * 2 > static_cast<int>(peers_.size()) + 1) { commitIndex_ = n; break; }
  }
}

void RaftNode::handleAppendEntriesResponse(int fromPeer, uint64_t term, bool success, uint64_t matchIndex) {
  std::vector<LogEntry> newlyCommitted;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (term > currentTerm_) { becomeFollower(term); return; }
    if (state_ != RaftState::LEADER || term != currentTerm_) return;
    PeerRpcState &ps = peerState_[static_cast<size_t>(fromPeer)];
    if (success) {
      ps.matchIndex = std::max(ps.matchIndex, matchIndex);
      ps.nextIndex = matchIndex + 1;
      advanceCommitIndex();
      while (lastApplied_ < commitIndex_) {
        ++lastApplied_;
        newlyCommitted.push_back(log_[lastApplied_]);
      }
    } else if (ps.nextIndex > 1) {
      ps.nextIndex -= 1; // back off and retry on the next heartbeat tick
    }
  }
  for (const LogEntry &e : newlyCommitted) { if (onCommit_) onCommit_(e); }
  if (!newlyCommitted.empty()) commitCv_.notify_all();
}

void RaftNode::receiverLoop(int peer) {
  for (;;) {
    std::vector<uint8_t> frame;
    try {
      frame = recvFrame(channel_, peer);
    } catch (...) {
      return;
    }
    if (frame.empty()) continue;
    Reader r{frame.data(), frame.size()};
    uint8_t tag = r.u8();
    switch (tag) {
      case kMsgShutdown:
        return;
      case kMsgRequestVote: {
        uint64_t term = r.u64();
        int candidateId = static_cast<int>(r.u32());
        uint64_t lastLogIndex = r.u64();
        uint64_t lastLogTerm = r.u64();
        handleRequestVote(peer, term, candidateId, lastLogIndex, lastLogTerm);
        break;
      }
      case kMsgRequestVoteResp: {
        uint64_t term = r.u64();
        bool voteGranted = r.u8() != 0;
        handleRequestVoteResponse(peer, term, voteGranted);
        break;
      }
      case kMsgAppendEntries: {
        uint64_t term = r.u64();
        int leaderIdIncoming = static_cast<int>(r.u32());
        uint64_t prevLogIndex = r.u64();
        uint64_t prevLogTerm = r.u64();
        uint64_t leaderCommit = r.u64();
        uint32_t numEntries = r.u32();
        std::vector<LogEntry> entries;
        entries.reserve(numEntries);
        for (uint32_t i = 0; i < numEntries; ++i) {
          uint64_t term_i = r.u64();
          std::string cmd = r.str();
          entries.push_back(LogEntry{term_i, prevLogIndex + 1 + i, cmd});
        }
        handleAppendEntries(peer, term, leaderIdIncoming, prevLogIndex, prevLogTerm, entries, leaderCommit);
        break;
      }
      case kMsgAppendEntriesResp: {
        uint64_t term = r.u64();
        bool success = r.u8() != 0;
        uint64_t matchIndex = r.u64();
        handleAppendEntriesResponse(peer, term, success, matchIndex);
        break;
      }
      default:
        break;
    }
  }
}

bool RaftNode::propose(const std::string &command, uint64_t timeout_ms) {
  uint64_t index;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != RaftState::LEADER) return false;
    index = log_.back().index + 1;
    log_.push_back(LogEntry{currentTerm_, index, command});
  }
  sendHeartbeats(); // replicate immediately rather than waiting for the next tick

  std::unique_lock<std::mutex> lock(mu_);
  return commitCv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return commitIndex_ >= index; });
}

RaftState RaftNode::state() const { std::lock_guard<std::mutex> lock(mu_); return state_; }
int RaftNode::leaderId() const { std::lock_guard<std::mutex> lock(mu_); return leaderId_; }
uint64_t RaftNode::currentTerm() const { std::lock_guard<std::mutex> lock(mu_); return currentTerm_; }
void RaftNode::onCommit(CommitCallback cb) { std::lock_guard<std::mutex> lock(mu_); onCommit_ = std::move(cb); }

} // namespace raft
