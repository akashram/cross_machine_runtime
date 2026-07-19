#pragma once
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
// Requires Linux + libfabric + an EFA-capable NIC. See rdma_transport.cpp
// for why this is a thin wrapper rather than exposing the full libfabric
// object graph — see networking/DESIGN.md §1.

// Opaque libfabric object handles, forward-declared so this header doesn't
// require <rdma/fabric.h> for callers that only hold a pointer (e.g.
// ring_allreduce swapping this in for networking::common::Channel later).
struct fi_info;
struct fid_fabric;
struct fid_domain;
struct fid_ep;
struct fid_cq;
struct fid_av;

class RdmaEndpoint {
public:
  ~RdmaEndpoint();

  // Resolve an EFA-capable provider and stand up fabric/domain/endpoint/
  // completion-queue/address-vector objects. Everything past this point
  // mirrors libfabric's fi_pingpong example structure (the closest thing
  // to a canonical usage reference for this API).
  bool init(const char *prov_name = "efa");

  // Passive side: bind the endpoint to `port` and wait for the active
  // side's first message (libfabric RDM endpoints are connectionless —
  // "listen" here means "insert the first peer address seen into the AV").
  bool listen(uint16_t port);

  // Active side: resolve `remote_addr:port` into an address-vector entry.
  bool connect(const char *remote_addr, uint16_t port);

  // Two-sided send/recv (blocking — polls the completion queue until the
  // corresponding fi_cq_read reports the operation's completion entry).
  ssize_t send(const void *buf, size_t len);
  ssize_t recv(void *buf, size_t len);

  void close();

private:
  fi_info *info_ = nullptr;
  fid_fabric *fabric_ = nullptr;
  fid_domain *domain_ = nullptr;
  fid_ep *ep_ = nullptr;
  fid_cq *cq_ = nullptr;
  fid_av *av_ = nullptr;
  uint64_t peer_addr_ = 0; // fi_addr_t, stored as raw bits to avoid the include here
};
