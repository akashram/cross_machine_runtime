#pragma once
#include <cstddef>

#include "channel.h"

// Binomial-tree all-reduce (sum): reduce-to-root followed by broadcast,
// O(log2(world_size)) round trips like halving_doubling (step 12), but
// each round trip moves the *entire* buffer rather than a shrinking/
// growing slice — bandwidth cost is O(count * log P), not the O(count)
// ring/halving-doubling achieve. Included specifically for the
// three-way comparison this step's README documents: at small message
// sizes where round-trip latency dominates, tree's simplicity can still
// win despite the worse bandwidth scaling.
//
// tree_reduce_to_root and tree_broadcast are exposed separately because
// collectives/ (step 14) reuses tree_broadcast as its Broadcast
// primitive — no reason to write binomial broadcast twice.
void tree_reduce_to_root(float *buf, size_t count, netcommon::Channel &channel, int root = 0);
void tree_broadcast(float *buf, size_t count, netcommon::Channel &channel, int root = 0);
void tree_allreduce(float *buf, size_t count, netcommon::Channel &channel);
