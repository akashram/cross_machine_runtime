#pragma once
#include <cstddef>
// TODO: implement on Linux with libfabric + EFA

struct RdmaRegion {
    void*    addr;
    size_t   len;
    uint64_t key;   // remote access key
};

// Register memory for one-sided RDMA access
bool rdma_register_memory(void* addr, size_t len, RdmaRegion& region);

// One-sided write: local_buf → remote.addr (no remote CPU involvement)
int rdma_write(const void* local_buf, size_t len, const RdmaRegion& remote);

// One-sided read: remote.addr → local_buf
int rdma_read(void* local_buf, size_t len, const RdmaRegion& remote);
