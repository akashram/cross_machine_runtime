//===- rdma_transport.cpp - libfabric/EFA two-sided transport -----------===//
//
// Structured after libfabric's fi_pingpong example: fi_getinfo to resolve
// an EFA-capable provider, fi_fabric/fi_domain/fi_endpoint to stand up the
// object graph, one completion queue shared by send and recv (order
// doesn't matter for two-sided messaging — the CQ entry's op_context
// tells us which), and one address-vector entry per peer since RDM
// endpoints are connectionless. See networking/DESIGN.md §1 for why
// ring_allreduce et al. are written against networking/common::Channel
// instead of this class directly — this is what an RdmaChannel adapter
// would wrap once there's EFA hardware to validate it against.
//
//===----------------------------------------------------------------------===//

#include "rdma_transport.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <cstdio>
#include <cstring>

namespace {
constexpr size_t kCqDepth = 128;

// Blocks until the completion queue reports one completion, retrying on
// FI_EAGAIN (queue empty). A real deployment would want a timeout here;
// left unbounded to match fi_pingpong's reference behavior and because
// this can't be exercised without EFA hardware to observe real failure
// modes against.
bool waitCompletion(fid_cq *cq) {
  fi_cq_entry entry;
  for (;;) {
    ssize_t ret = fi_cq_read(cq, &entry, 1);
    if (ret > 0) return true;
    if (ret == -FI_EAGAIN) continue;
    fi_cq_err_entry err{};
    fi_cq_readerr(cq, &err, 0);
    std::fprintf(stderr, "rdma_transport: cq error: %s\n", fi_strerror(err.err));
    return false;
  }
}
} // namespace

RdmaEndpoint::~RdmaEndpoint() { close(); }

bool RdmaEndpoint::init(const char *prov_name) {
  fi_info *hints = fi_allocinfo();
  hints->ep_attr->type = FI_EP_RDM; // reliable datagram — matches EFA's SRD transport
  hints->caps = FI_MSG;
  hints->mode = FI_CONTEXT;
  hints->fabric_attr->prov_name = strdup(prov_name);

  int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info_);
  fi_freeinfo(hints);
  if (rc != 0) {
    std::fprintf(stderr, "rdma_transport: fi_getinfo failed: %s\n", fi_strerror(-rc));
    return false;
  }

  if (fi_fabric(info_->fabric_attr, &fabric_, nullptr) != 0) return false;
  if (fi_domain(fabric_, info_, &domain_, nullptr) != 0) return false;

  fi_cq_attr cqAttr{};
  cqAttr.size = kCqDepth;
  cqAttr.format = FI_CQ_FORMAT_CONTEXT;
  if (fi_cq_open(domain_, &cqAttr, &cq_, nullptr) != 0) return false;

  fi_av_attr avAttr{};
  avAttr.type = FI_AV_TABLE;
  if (fi_av_open(domain_, &avAttr, &av_, nullptr) != 0) return false;

  if (fi_endpoint(domain_, info_, &ep_, nullptr) != 0) return false;
  fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV);
  fi_ep_bind(ep_, &av_->fid, 0);
  if (fi_enable(ep_) != 0) return false;

  return true;
}

bool RdmaEndpoint::listen(uint16_t /*port*/) {
  // RDM endpoints have no listen()/accept() — the first `recv` posted
  // against a not-yet-inserted peer address will fail until `connect()`
  // (on either side) inserts it into the AV. This method exists so the
  // server/client CLI shape matches efa_setup/efa_bench.sh's; the real
  // "accept" is the AV insertion that happens in `connect`.
  return ep_ != nullptr;
}

bool RdmaEndpoint::connect(const char *remote_addr, uint16_t port) {
  fi_info *hints = fi_allocinfo();
  hints->ep_attr->type = FI_EP_RDM;
  fi_info *remoteInfo = nullptr;
  std::string node = remote_addr;
  std::string service = std::to_string(port);
  if (fi_getinfo(FI_VERSION(1, 18), node.c_str(), service.c_str(), 0, hints, &remoteInfo) != 0) {
    fi_freeinfo(hints);
    return false;
  }
  fi_freeinfo(hints);

  fi_addr_t addr = FI_ADDR_UNSPEC;
  int inserted = fi_av_insert(av_, remoteInfo->dest_addr, 1, &addr, 0, nullptr);
  fi_freeinfo(remoteInfo);
  if (inserted != 1) return false;

  peer_addr_ = static_cast<uint64_t>(addr);
  return true;
}

ssize_t RdmaEndpoint::send(const void *buf, size_t len) {
  int rc = fi_send(ep_, buf, len, nullptr, static_cast<fi_addr_t>(peer_addr_), nullptr);
  if (rc != 0) return -1;
  return waitCompletion(cq_) ? static_cast<ssize_t>(len) : -1;
}

ssize_t RdmaEndpoint::recv(void *buf, size_t len) {
  int rc = fi_recv(ep_, buf, len, nullptr, static_cast<fi_addr_t>(peer_addr_), nullptr);
  if (rc != 0) return -1;
  return waitCompletion(cq_) ? static_cast<ssize_t>(len) : -1;
}

void RdmaEndpoint::close() {
  if (ep_) { fi_close(&ep_->fid); ep_ = nullptr; }
  if (av_) { fi_close(&av_->fid); av_ = nullptr; }
  if (cq_) { fi_close(&cq_->fid); cq_ = nullptr; }
  if (domain_) { fi_close(&domain_->fid); domain_ = nullptr; }
  if (fabric_) { fi_close(&fabric_->fid); fabric_ = nullptr; }
  if (info_) { fi_freeinfo(info_); info_ = nullptr; }
}
