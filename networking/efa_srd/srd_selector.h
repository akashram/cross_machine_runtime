//===- srd_selector.h - SRD vs RC transport selection --------------------===//
//
// Portable — no hardware or libfabric dependency. EFA offers two wire
// transports: SRD (Scalable Reliable Datagram — reliable, but delivers
// out of order, and one queue pair (QP) can talk to many peers) and RC
// (Reliable Connected — in-order, but one QP per peer, so an all-to-all
// pattern needs O(N) QPs per node). This header is the decision this
// project's transport layer makes between them, extracted as pure logic
// so it's testable without an EFA NIC — see srd_selector_test.cpp.
//
//===----------------------------------------------------------------------===//
#pragma once
#include <cstddef>

enum class EfaTransport { SRD, RC };

struct WorkloadProfile {
  int num_peers;          // distinct peers this endpoint talks to
  size_t message_size;    // typical message size, bytes
  bool ordering_required; // does the caller need in-order delivery from the
                           // transport itself, or does it reorder at a
                           // higher layer (e.g. sequence numbers in Raft's
                           // AppendEntries, or the fixed round structure of
                           // ring all-reduce)?
};

// SRD wins whenever QP scalability matters more than in-order delivery:
// many peers (collectives, all-to-all MoE dispatch, gRPC-style fan-out)
// where reliable-but-unordered is fine because the caller already has a
// sequencing mechanism. RC wins for a small, fixed set of peers with a
// protocol that assumes wire-order delivery (e.g. a raw two-sided
// send/recv stream with no application-level sequence numbers).
//
// The actual crossover point (how many peers before SRD's QP-scaling
// advantage outweighs RC's simplicity) needs the latency/throughput
// numbers this step's README marks TODO — this function encodes the
// *shape* of the decision, which doesn't change once those numbers land,
// only the threshold constant does.
EfaTransport select_transport(const WorkloadProfile &profile);
