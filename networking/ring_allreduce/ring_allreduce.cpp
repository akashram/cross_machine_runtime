//===- ring_allreduce.cpp - Step 11 implementation ------------------------===//
//
// Deadlock avoidance: each rank's ring neighbors are two *different*
// peers (right = rank+1, left = rank-1), so on a real multi-node
// deployment (separate sockets, no sharing) send/recv order wouldn't
// matter. It matters here because networking/common::TcpChannel shares
// one bidirectional socket per peer pair (see channel.h) — for world_size
// == 2, "left" and "right" are the *same* peer, so send-then-recv on both
// ends deadlocks once a chunk exceeds the kernel socket buffer. The fix
// is the standard ring-allreduce ordering trick: even ranks send before
// receiving, odd ranks receive before sending. Trace it around the ring:
// rank r (even) sends to r+1 (odd), which is already waiting in recv()
// first — so r's send never blocks. r+1 then sends to r+2 (even), which
// is mid-send to r+3 but will reach recv() right after — no rank is ever
// waiting on a peer that's waiting on it. This generalizes correctly to
// world_size == 2 as well (rank 0 even, rank 1 odd).
//
//===----------------------------------------------------------------------===//

#include "ring_allreduce.h"

#include <algorithm>
#include <vector>

using netcommon::Channel;

namespace {

struct ChunkLayout {
  std::vector<size_t> start, size;
};

ChunkLayout computeChunks(size_t count, int n) {
  ChunkLayout layout;
  layout.start.resize(n);
  layout.size.resize(n);
  size_t base = count / static_cast<size_t>(n);
  size_t remainder = count % static_cast<size_t>(n);
  size_t offset = 0;
  for (int i = 0; i < n; ++i) {
    layout.start[i] = offset;
    layout.size[i] = base + (static_cast<size_t>(i) < remainder ? 1 : 0);
    offset += layout.size[i];
  }
  return layout;
}

} // namespace

void ring_reduce_scatter(float *buf, size_t count, Channel &channel) {
  int rank = channel.rank();
  int n = channel.world_size();
  if (n <= 1) return;

  int right = (rank + 1) % n;
  int left = (rank - 1 + n) % n;
  bool sendFirst = (rank % 2 == 0);
  ChunkLayout layout = computeChunks(count, n);

  size_t maxChunkBytes = *std::max_element(layout.size.begin(), layout.size.end()) * sizeof(float);
  std::vector<float> recvTmp(maxChunkBytes / sizeof(float));

  // After step s, this rank has accumulated s+1 ranks' contributions into
  // chunk (rank - s - 1) mod n.
  for (int step = 0; step < n - 1; ++step) {
    int sendIdx = (rank - step + n) % n;
    int recvIdx = (rank - step - 1 + n) % n;
    auto doSend = [&] { channel.send(right, buf + layout.start[sendIdx], layout.size[sendIdx] * sizeof(float)); };
    auto doRecv = [&] { channel.recv(left, recvTmp.data(), layout.size[recvIdx] * sizeof(float)); };
    if (sendFirst) { doSend(); doRecv(); } else { doRecv(); doSend(); }
    for (size_t i = 0; i < layout.size[recvIdx]; ++i)
      buf[layout.start[recvIdx] + i] += recvTmp[i];
  }
}

void ring_all_gather(float *buf, size_t count, Channel &channel) {
  int rank = channel.rank();
  int n = channel.world_size();
  if (n <= 1) return;

  int right = (rank + 1) % n;
  int left = (rank - 1 + n) % n;
  bool sendFirst = (rank % 2 == 0);
  ChunkLayout layout = computeChunks(count, n);

  // Circulate each rank's already-correct chunk around the ring —
  // starting point is rank's own chunk (index `rank`), no accumulation,
  // just overwrite. Index formula is offset by one round from
  // ring_reduce_scatter's — the first chunk forwarded here is the one
  // reduce_scatter *just finished* reducing (chunk (rank-1) mod n, this
  // rank's own final chunk from that phase), not the one it's about to
  // start reducing.
  for (int step = 0; step < n - 1; ++step) {
    int sendIdx = (rank - step + 1 + n) % n;
    int recvIdx = (rank - step + n) % n;
    auto doSend = [&] { channel.send(right, buf + layout.start[sendIdx], layout.size[sendIdx] * sizeof(float)); };
    auto doRecv = [&] { channel.recv(left, buf + layout.start[recvIdx], layout.size[recvIdx] * sizeof(float)); };
    if (sendFirst) { doSend(); doRecv(); } else { doRecv(); doSend(); }
  }
}

void ring_allreduce(float *buf, size_t count, Channel &channel) {
  if (channel.world_size() <= 1) return;
  ring_reduce_scatter(buf, count, channel);
  ring_all_gather(buf, count, channel);
}
