#pragma once
#include <cstddef>

#include "channel.h"

// Ring reduce-scatter (step 11's phase 1, exposed standalone for
// collectives/ step 14): after this call, rank r's OWNED slice —
// chunk index `(r+1) % world_size` (not `r`; that offset falls out of
// the ring's direction and isn't worth an extra rotation step to hide —
// see ring_allreduce.cpp) — holds the fully-reduced sum of every rank's
// contribution to that slice; every other slice is partially-reduced
// garbage. Chunk `i` spans `[i*count/N, (i+1)*count/N)`. n-1 rounds.
void ring_reduce_scatter(float *buf, size_t count, netcommon::Channel &channel);

// Ring all-gather (step 11's phase 2, exposed standalone for
// collectives/): circulates each rank's already-correct chunk
// `(rank+1) % world_size` (matching ring_reduce_scatter's output
// convention above) around the ring so every rank ends up with the full
// buffer. n-1 rounds. Correct to call directly (without reduce_scatter
// first) whenever that specific chunk already holds the value every rank
// should end up seeing there — that's collectives::AllGather's contract,
// not just ring_allreduce's internal phase 2.
void ring_all_gather(float *buf, size_t count, netcommon::Channel &channel);

// In-place ring all-reduce (sum) on buf[count] floats across every rank
// reachable via `channel`: ring_reduce_scatter then ring_all_gather,
// 2*(world_size-1) rounds total. Bandwidth-optimal: total data moved per
// rank is ~2*(N-1)/N * count*sizeof(float) — approaches 2x the buffer
// size regardless of N for large N, unlike a naive gather-to-one/reduce/
// broadcast which moves O(N) at the root. See ring_allreduce.cpp for the
// deadlock-avoidance ordering this needs on a shared-socket transport.
void ring_allreduce(float *buf, size_t count, netcommon::Channel &channel);
