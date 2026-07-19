#include "collectives.h"

#include "ring_allreduce.h"
#include "tree_allreduce.h"

#include <algorithm>

namespace collectives {

void Broadcast(float *buf, size_t count, netcommon::Channel &channel, int root) {
  tree_broadcast(buf, count, channel, root);
}

void ReduceScatter(float *buf, size_t count, netcommon::Channel &channel) {
  ring_reduce_scatter(buf, count, channel);
}

void AllGather(const float *send_buf, size_t send_count, float *recv_buf,
                netcommon::Channel &channel) {
  int rank = channel.rank();
  int n = channel.world_size();
  // Place this rank's own contribution at slot (rank+1)%n — not `rank` —
  // before circulating: that's the chunk ring_all_gather trusts is
  // already correct and sends out first (see ring_allreduce.h), the same
  // convention ring_reduce_scatter's output uses.
  size_t slot = static_cast<size_t>((rank + 1) % n);
  std::copy(send_buf, send_buf + send_count, recv_buf + slot * send_count);
  ring_all_gather(recv_buf, send_count * static_cast<size_t>(n), channel);
}

} // namespace collectives
