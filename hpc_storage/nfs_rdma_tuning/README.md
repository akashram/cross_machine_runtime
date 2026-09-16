# nfs_rdma_tuning

**Status: literature/vendor-doc-grounded written analysis. No code — see
"Definition of done" below for why.**

## What this covers

PLAN.md Phase 21 step 5: applying `networking/rdma_v1`'s and
`networking/nic_deep_dive`'s existing RDMA/NIC primitives to STORAGE
traffic specifically — NFS over RDMA (NFSoRDMA) / RoCE tuning for a
GPU-to-storage path — contrasted with the small, latency-sensitive
collective messages that phase's original RDMA work targets. Written
analysis, no new local measurement (no RDMA-capable NIC or NFSoRDMA mount
exists locally — see `networking/rdma_v1/README.md`'s own hardware-gated
status, which this step inherits rather than duplicates).

## What changes: collective messages vs. bulk storage traffic

`networking/rdma_v1/rdma_transport.h` and `networking/nic_deep_dive`'s
material were both written for `networking/`'s Phase 5 use case:
**small, latency-sensitive collective messages** (gradient all-reduce
chunks, control-plane RPCs) where the tuning goal is minimizing
per-message LATENCY. Storage traffic for AI training — bulk dataset
reads, checkpoint writes at the sizes `checkpoint_burst_capacity`
(step 6) measures — is a different tuning problem: the goal is maximizing
sustained THROUGHPUT over large transfers, not minimizing the latency of
any one small message. The same underlying RDMA transport mechanics
apply, but the knobs that matter shift:

- **Queue depth**: collective tuning favors a MODERATE queue depth (deep
  enough to hide latency, shallow enough that a straggler doesn't
  backlog the whole collective — see `networking/backpressure`'s
  credit-based design for the general shape of this tradeoff). Bulk
  storage tuning favors a DEEP queue depth — more in-flight requests
  directly increases achieved bandwidth on a high-bandwidth-delay-product
  path, since throughput is bounded by `queue_depth * request_size /
  RTT` until the link saturates.
- **MTU / jumbo frames**: largely irrelevant for the small collective
  messages `networking/`'s existing work optimizes (a gradient chunk
  often fits in one MTU regardless), but directly load-bearing for bulk
  storage transfer — larger frames (9000-byte jumbo vs. 1500-byte
  standard Ethernet MTU) reduce per-packet header overhead and interrupt
  rate proportionally to transfer size, a real percentage-of-throughput
  effect at storage-transfer scale that doesn't show up at
  collective-message scale.
- **ECN/PFC congestion control**: RoCEv2's lossless-fabric requirement
  (Priority Flow Control, PFC) exists to prevent RDMA's in-order,
  no-retransmit-friendly transport from stalling on packet loss — this
  matters MORE for storage bulk transfer than for small collective
  messages precisely because a bulk transfer holds a PFC-paused queue
  occupied far longer, and PFC head-of-line blocking on a shared switch
  fabric can stall unrelated flows (a well-documented RoCEv2 deployment
  concern) — a failure mode that barely surfaces at collective-message
  message sizes but is a real operational risk at storage-transfer scale.

## The GPU-to-storage path specifically

`distributed_training/gpudirect_storage`'s cuFile client bypasses host
CPU staging for the storage-to-GPU-HBM leg of a read; NFSoRDMA is the
analogous idea one layer up the stack — bypassing TCP's kernel-socket
staging for the storage-to-host-memory leg when GDS itself isn't
available or when the storage backend is presented over NFS rather than
a raw block/cuFile path. The two compose: NFSoRDMA gets bytes from the
storage server to host (or, with GDS-aware NFS clients, more directly)
efficiently: bypassing double-buffering, while cuFile/GDS gets bytes from
there to GPU HBM without a second host-memory bounce.

## Sources

RFC 8267 (NFS over RDMA transport), RoCEv2 (InfiniBand Trade Association)
and its PFC/ECN congestion-control literature — see `READING_LIST.md`'s
Phase 21 section for full citations.

## Definition of done

Same "literature-grounded, no local measurement possible" status as
`storage_comparison` (step 3) and `vast_dase_gds` (step 4) — no RDMA NIC
or NFSoRDMA mount exists on this Mac, inheriting
`networking/rdma_v1/README.md`'s own hardware-gated status rather than
duplicating a new gap.
