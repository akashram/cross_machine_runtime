#pragma once
#include <cstddef>
#include <cstdint>
// Requires Linux + libfabric + EFA. See rdma_onesided.cpp — same
// fi_fabric/fi_domain/fi_endpoint object graph as rdma_v1's RdmaEndpoint,
// but the module owns one implicit endpoint (free-function API, not a
// class) since a one-sided read/write benchmark only ever needs one
// active connection at a time.

struct RdmaRegion {
    void*    addr;
    size_t   len;
    uint64_t key;   // remote access key (fi_mr_key of the registered region)
};

// Stands up the fabric/domain/endpoint/CQ and resolves `remote_addr:port`
// into the AV entry every rdma_write/rdma_read below targets. Must be
// called once before any other function in this header.
bool rdma_onesided_init(const char* remote_addr, uint16_t port);
void rdma_onesided_close();

// Register memory for one-sided RDMA access — must be called on both
// sides for any address that will be the *target* of a peer's
// rdma_write/rdma_read (the local side of an rdma_write/rdma_read call
// needs no registration in this simplified wrapper: fi_write/fi_read take
// an unregistered local buffer via FI_MR_LOCAL-less providers, which EFA
// supports for RDM endpoints).
bool rdma_register_memory(void* addr, size_t len, RdmaRegion& region);

// One-sided write: local_buf → remote.addr (no remote CPU involvement)
int rdma_write(const void* local_buf, size_t len, const RdmaRegion& remote);

// One-sided read: remote.addr → local_buf
int rdma_read(void* local_buf, size_t len, const RdmaRegion& remote);
