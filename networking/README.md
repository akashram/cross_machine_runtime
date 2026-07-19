# Phase 5: Distributed Layer + Networking

**Status: code-complete (25/25 steps). 12 steps built and run locally
(Mac) — see `DESIGN.md` for why most of this phase doesn't actually need
EFA hardware. Remaining 13 steps are real, complete code gated behind
Linux-only kernel APIs, specific NICs, external libraries, GPU hardware,
or a Java toolchain — see each step's own README.**

## Overview
Full distributed networking stack: EFA/RDMA transport, PTP clock sync,
gRPC control plane, AF_XDP kernel bypass, ring/tree/halving-doubling all-reduce,
Raft consensus, and chaos engineering harness. Built on a shared portable
`common/Channel` transport (see `DESIGN.md` §1) wherever the algorithm
itself doesn't need EFA specifically.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| — | common | shared Channel/TcpChannel substrate | **built + run** |
| 1 | efa_setup | EFA install, fi_pingpong baseline | script complete, EFA-gated |
| 2 | rdma_v1 | libfabric two-sided send/recv + TCP baseline | TCP baseline **built + run**; EFA half gated |
| 3 | rdma_onesided | fi_read/fi_write | code-complete, EFA-gated |
| 4 | efa_srd | SRD vs RC transport selection | **built + run** |
| 5 | ptp | IEEE 1588 clock sync | code-complete, Linux-gated |
| 6 | grpc_control | gRPC + protobuf control plane | code-complete, gRPC-gated |
| 7 | flatbuffers_data | zero-copy serialization for data plane | code-complete, FlatBuffers-gated |
| 8 | af_xdp | AF_XDP kernel bypass | code-complete, Linux-gated |
| 9 | userspace_net | full userspace networking stack | code-complete, Linux-gated |
| 10 | nic_deep_dive | NIC descriptor rings, RSS, PFC/ECN | script complete, Linux-gated |
| 11 | ring_allreduce | ring all-reduce from scratch | **built + run** |
| 12 | halving_doubling | recursive halving-doubling all-reduce | **built + run** |
| 13 | tree_allreduce | tree all-reduce | **built + run** |
| 14 | collectives | broadcast, reduce-scatter, all-gather | **built + run** |
| 15 | nccl_tuning | NCCL_ALGO/PROTO/BUFFSIZE tuning | script complete, GPU-gated |
| 16 | topo_scheduler | topology-aware scheduler | **built + run** |
| 17 | vector_clocks | Lamport timestamps + vector clocks | **built + run** |
| 18 | chandy_lamport | distributed global state capture | **built + run** |
| 19 | raft | Raft consensus from scratch | **built + run** |
| 20 | tla_raft | TLA+ spec + TLC model checker | spec written, TLC deferred |
| 21 | backpressure | token bucket + load shedding | **built + run** |
| 22 | hedged_requests | hedged requests for tail latency | **built + run** |
| 23 | multitenancy | resource quotas + fair scheduling | **built + run** |
| 24 | chaos | tc netem injection harness | scripts complete, Linux-gated |
| 25 | tla_collective | TLA+ for all-reduce protocol | spec written, TLC deferred |

## Hardware notes
- EFA steps (1-4 partial, plus real cross-node numbers everywhere): 2×
  p4d.24xlarge in VPC placement group with EFA-enabled NICs
- Everything marked **built + run** above: validated on this Mac already,
  no hardware needed for correctness — only for real-network performance
  numbers (each step's README has a Results table marked TODO for those)
