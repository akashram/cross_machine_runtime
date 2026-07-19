//===- collectives.h - Complete collective communication library --------===//
//
// Not new algorithms — this step assembles the primitives steps 11-13
// already built into the standard MPI-style collective vocabulary, and
// exposes them independently (a caller wanting only Broadcast shouldn't
// need to know it happens to reuse the tree implementation). See
// networking/DESIGN.md for why the algorithms live where they do
// (ring/tree own the primitives; this is the assembly/naming layer).
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>

#include "channel.h"

namespace collectives {

// Every rank ends up with `root`'s buffer. Reuses tree_broadcast
// (tree_allreduce/, step 13) — binomial tree, O(log P) round trips.
void Broadcast(float *buf, size_t count, netcommon::Channel &channel, int root = 0);

// Rank r's owned slice — chunk index `(r+1) % world_size`, i.e.
// `[((r+1)%N)*count/N, ((r+1)%N + 1)*count/N)` — ends up holding the
// element-wise sum of every rank's contribution to that slice. The
// off-by-one chunk numbering (not simply `r`) comes directly from
// ring_reduce_scatter's ring direction; see its doc comment. Reuses
// ring_reduce_scatter (ring_allreduce/, step 11) — bandwidth-optimal.
void ReduceScatter(float *buf, size_t count, netcommon::Channel &channel);

// Every rank ends up with the concatenation of every rank's
// `send_count`-sized contribution, placed at slot `(r+1) % world_size`
// (matching ReduceScatter's chunk numbering above — this AllGather is
// literally what running ReduceScatter's all-gather half alone looks
// like, so the two share one placement convention): rank r's input
// occupies `[((r+1)%N)*send_count, ((r+1)%N + 1)*send_count)` of
// `recv_buf` on every rank. Reuses ring_all_gather (ring_allreduce/, step 11).
void AllGather(const float *send_buf, size_t send_count, float *recv_buf,
                netcommon::Channel &channel);

} // namespace collectives
