#pragma once
#include <cstddef>
// TODO: implement on Linux with multi-node setup

// In-place ring all-reduce on buf[count] floats across all ranks.
// rank: this process rank; world_size: total processes
void ring_allreduce(float* buf, size_t count, int rank, int world_size);
