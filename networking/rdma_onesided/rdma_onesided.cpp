//===- rdma_onesided.cpp - fi_read/fi_write one-sided RDMA ---------------===//
//
// One-sided ops are the whole point of RDMA over two-sided send/recv: the
// remote CPU never runs a matching recv() — the NIC writes (or reads)
// directly into (from) a pre-registered remote memory region. That's why
// `rdma_register_memory` has to run on *both* peers before either side
// calls rdma_write/rdma_read: the target region's `key` has to already
// exist in the target's memory-registration table for the initiator's
// fi_write/fi_read to be allowed to touch it.
//
//===----------------------------------------------------------------------===//

#include "rdma_onesided.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <cstdio>
#include <cstring>

namespace {

struct OnesidedState {
  fi_info *info = nullptr;
  fid_fabric *fabric = nullptr;
  fid_domain *domain = nullptr;
  fid_ep *ep = nullptr;
  fid_cq *cq = nullptr;
  fid_av *av = nullptr;
  fi_addr_t peer = FI_ADDR_UNSPEC;
} g;

bool waitCompletion() {
  fi_cq_entry entry;
  for (;;) {
    ssize_t ret = fi_cq_read(g.cq, &entry, 1);
    if (ret > 0) return true;
    if (ret == -FI_EAGAIN) continue;
    fi_cq_err_entry err{};
    fi_cq_readerr(g.cq, &err, 0);
    std::fprintf(stderr, "rdma_onesided: cq error: %s\n", fi_strerror(err.err));
    return false;
  }
}

} // namespace

bool rdma_onesided_init(const char *remote_addr, uint16_t port) {
  fi_info *hints = fi_allocinfo();
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps = FI_MSG | FI_RMA; // RMA capability is what makes fi_write/fi_read valid
  hints->mode = FI_CONTEXT;
  hints->fabric_attr->prov_name = strdup("efa");

  std::string service = std::to_string(port);
  if (fi_getinfo(FI_VERSION(1, 18), remote_addr, service.c_str(), 0, hints, &g.info) != 0) {
    fi_freeinfo(hints);
    return false;
  }
  fi_freeinfo(hints);

  if (fi_fabric(g.info->fabric_attr, &g.fabric, nullptr) != 0) return false;
  if (fi_domain(g.fabric, g.info, &g.domain, nullptr) != 0) return false;

  fi_cq_attr cqAttr{};
  cqAttr.size = 128;
  cqAttr.format = FI_CQ_FORMAT_CONTEXT;
  if (fi_cq_open(g.domain, &cqAttr, &g.cq, nullptr) != 0) return false;

  fi_av_attr avAttr{};
  avAttr.type = FI_AV_TABLE;
  if (fi_av_open(g.domain, &avAttr, &g.av, nullptr) != 0) return false;

  if (fi_endpoint(g.domain, g.info, &g.ep, nullptr) != 0) return false;
  fi_ep_bind(g.ep, &g.cq->fid, FI_SEND | FI_RECV | FI_READ | FI_WRITE);
  fi_ep_bind(g.ep, &g.av->fid, 0);
  if (fi_enable(g.ep) != 0) return false;

  return fi_av_insert(g.av, g.info->dest_addr, 1, &g.peer, 0, nullptr) == 1;
}

void rdma_onesided_close() {
  if (g.ep) { fi_close(&g.ep->fid); g.ep = nullptr; }
  if (g.av) { fi_close(&g.av->fid); g.av = nullptr; }
  if (g.cq) { fi_close(&g.cq->fid); g.cq = nullptr; }
  if (g.domain) { fi_close(&g.domain->fid); g.domain = nullptr; }
  if (g.fabric) { fi_close(&g.fabric->fid); g.fabric = nullptr; }
  if (g.info) { fi_freeinfo(g.info); g.info = nullptr; }
}

bool rdma_register_memory(void *addr, size_t len, RdmaRegion &region) {
  fid_mr *mr = nullptr;
  uint64_t requestedKey = reinterpret_cast<uint64_t>(addr); // provider may reassign; read back below
  int rc = fi_mr_reg(g.domain, addr, len, FI_REMOTE_READ | FI_REMOTE_WRITE, 0,
                      requestedKey, 0, &mr, nullptr);
  if (rc != 0) {
    std::fprintf(stderr, "rdma_onesided: fi_mr_reg failed: %s\n", fi_strerror(-rc));
    return false;
  }
  region.addr = addr;
  region.len = len;
  region.key = fi_mr_key(mr);
  // Deliberately leaking `mr` for the lifetime of the process — a
  // benchmark-scoped simplification (see header comment); a long-lived
  // service would track and fi_close() it on teardown.
  return true;
}

int rdma_write(const void *local_buf, size_t len, const RdmaRegion &remote) {
  int rc = fi_write(g.ep, local_buf, len, nullptr, g.peer,
                     reinterpret_cast<uint64_t>(remote.addr), remote.key, nullptr);
  if (rc != 0) return -1;
  return waitCompletion() ? static_cast<int>(len) : -1;
}

int rdma_read(void *local_buf, size_t len, const RdmaRegion &remote) {
  int rc = fi_read(g.ep, local_buf, len, nullptr, g.peer,
                    reinterpret_cast<uint64_t>(remote.addr), remote.key, nullptr);
  if (rc != 0) return -1;
  return waitCompletion() ? static_cast<int>(len) : -1;
}
