//===- halving_doubling.cpp - Step 12 implementation ----------------------===//
//
// Reduce-scatter (recursive halving): at each step, split the currently-
// owned range in half; the lower-ranked side of the pair keeps the lower
// half and sends the upper half (and vice versa), then reduces the
// received half into its kept half. Because "lower rank keeps lower half"
// is applied consistently at every step, after log2(P) steps rank r ends
// up owning exactly the r-th 1/P slice of the original buffer — the same
// partition ring_allreduce uses, which is why all-gather below can share
// its chunk-boundary math.
//
// All-gather (recursive doubling) is the exact mirror: start with the
// small owned slice and double its extent each step by exchanging with a
// partner and appending/prepending their slice, until every rank holds
// the full buffer.
//
//===----------------------------------------------------------------------===//

#include "halving_doubling.h"
#include "ring_allreduce.h"

#include <vector>

using netcommon::Channel;

namespace {
bool isPowerOfTwo(int n) { return n > 0 && (n & (n - 1)) == 0; }
} // namespace

void halving_doubling_allreduce(float *buf, size_t count, Channel &channel) {
  int rank = channel.rank();
  int p = channel.world_size();
  if (p <= 1) return;
  if (!isPowerOfTwo(p)) {
    ring_allreduce(buf, count, channel); // documented fallback — see header
    return;
  }

  // --- Phase 1: recursive-halving reduce-scatter ---
  size_t myOffset = 0;
  size_t myCount = count;
  for (int mask = p / 2; mask > 0; mask >>= 1) {
    int partner = rank ^ mask;
    bool keepLower = (rank & mask) == 0; // lower rank in the pair keeps the lower half

    size_t lowerCount = myCount / 2;
    size_t upperCount = myCount - lowerCount;
    size_t sendCount, recvCount, sendOffset, recvOffset;
    if (keepLower) {
      sendCount = upperCount;
      sendOffset = myOffset + lowerCount;
      recvCount = lowerCount;
      recvOffset = myOffset;
    } else {
      sendCount = lowerCount;
      sendOffset = myOffset;
      recvCount = upperCount;
      recvOffset = myOffset + lowerCount;
    }

    std::vector<float> recvTmp(recvCount);
    // Deadlock avoidance: same shared-socket concern as ring_allreduce —
    // exactly one of the two partners must send first. Using the same
    // rank-parity-independent rule ring_allreduce documents doesn't apply
    // directly here (this is pairwise, not ring-adjacent), so instead use
    // the pair's natural order: lower rank sends first, matching
    // networking/common::Channel's own deadlock-safe-ordering convention
    // (see channel.h / channel_test.cpp).
    if (rank < partner) {
      channel.send(partner, buf + sendOffset, sendCount * sizeof(float));
      channel.recv(partner, recvTmp.data(), recvCount * sizeof(float));
    } else {
      channel.recv(partner, recvTmp.data(), recvCount * sizeof(float));
      channel.send(partner, buf + sendOffset, sendCount * sizeof(float));
    }
    for (size_t i = 0; i < recvCount; ++i) buf[recvOffset + i] += recvTmp[i];

    myOffset = recvOffset;
    myCount = recvCount;
  }
  // myCount == count / p now; myOffset is this rank's slice of the r-th
  // 1/p partition (same partition ring_allreduce.cpp computes explicitly).

  // --- Phase 2: recursive-doubling all-gather ---
  for (int mask = 1; mask < p; mask <<= 1) {
    int partner = rank ^ mask;
    size_t partnerOffset;
    if (rank < partner) {
      partnerOffset = myOffset + myCount; // partner's slice sits right after mine
    } else {
      partnerOffset = myOffset - myCount; // partner's slice sits right before mine
      myOffset = partnerOffset;
    }

    if (rank < partner) {
      channel.send(partner, buf + myOffset, myCount * sizeof(float));
      channel.recv(partner, buf + partnerOffset, myCount * sizeof(float));
    } else {
      channel.recv(partner, buf + partnerOffset, myCount * sizeof(float));
      channel.send(partner, buf + (myOffset + myCount), myCount * sizeof(float));
    }
    myCount *= 2;
  }
}
