//===- tree_allreduce.cpp - Step 13 implementation ------------------------===//
//
// Both functions use rank *relative to root* (`(rank - root + P) % P`) so
// the same bit-mask logic works for any root, not just 0. Unlike
// ring_allreduce/halving_doubling, no explicit send-before-recv ordering
// trick is needed: at every round, each pair's roles are structurally
// asymmetric (whoever's bit is set sends, the other unconditionally
// receives — there's no round where both sides of a pair might try to
// send, so nothing to accidentally get backwards).
//
//===----------------------------------------------------------------------===//

#include "tree_allreduce.h"

#include <vector>

using netcommon::Channel;

namespace {
int toActual(int relative, int root, int p) { return (relative + root) % p; }
} // namespace

void tree_reduce_to_root(float *buf, size_t count, Channel &channel, int root) {
  int rank = channel.rank();
  int p = channel.world_size();
  if (p <= 1) return;
  int relative = (rank - root + p) % p;

  std::vector<float> tmp(count);
  for (int mask = 1; mask < p; mask <<= 1) {
    if (relative & mask) {
      int partner = toActual(relative & ~mask, root, p);
      channel.send(partner, buf, count * sizeof(float));
      break; // sent everything I have accumulated — my part is done
    }
    int peerRelative = relative | mask;
    if (peerRelative < p) {
      int partner = toActual(peerRelative, root, p);
      channel.recv(partner, tmp.data(), count * sizeof(float));
      for (size_t i = 0; i < count; ++i) buf[i] += tmp[i];
    }
  }
}

void tree_broadcast(float *buf, size_t count, Channel &channel, int root) {
  int rank = channel.rank();
  int p = channel.world_size();
  if (p <= 1) return;
  int relative = (rank - root + p) % p;

  for (int mask = 1; mask < p; mask <<= 1) {
    if (relative < mask) {
      int peerRelative = relative + mask;
      if (peerRelative < p) channel.send(toActual(peerRelative, root, p), buf, count * sizeof(float));
    } else if (relative < 2 * mask) {
      int peerRelative = relative - mask;
      channel.recv(toActual(peerRelative, root, p), buf, count * sizeof(float));
    }
  }
}

void tree_allreduce(float *buf, size_t count, Channel &channel) {
  tree_reduce_to_root(buf, count, channel, /*root=*/0);
  tree_broadcast(buf, count, channel, /*root=*/0);
}
