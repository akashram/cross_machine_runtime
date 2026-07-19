#pragma once
#include <cstddef>

#include "channel.h"

// Recursive halving-doubling all-reduce (sum), a.k.a. Rabenseifner's
// algorithm: log2(world_size) steps of recursive-halving reduce-scatter
// (each step, exchange with a partner at halving distance, keep/reduce
// one half, forward the other) followed by log2(world_size) steps of
// recursive-doubling all-gather (the mirror image). Same bandwidth-optimal
// total data movement as ring_allreduce (step 11) but O(log N) network
// round trips instead of O(N) — wins for small messages where per-round-trip
// latency dominates over bandwidth; ring wins once messages are large
// enough that bandwidth dominates. See README.md for the crossover.
//
// Requires world_size to be a power of 2 (the classic algorithm's
// precondition); falls back to ring_allreduce otherwise rather than
// implementing the more complex non-power-of-2 variant (Rabenseifner's
// paper handles it by folding extra ranks into their neighbors first —
// noted as future work, not needed until a real cluster size demands it).
void halving_doubling_allreduce(float *buf, size_t count, netcommon::Channel &channel);
