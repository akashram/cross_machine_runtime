# Phase 5: Distributed Layer + Networking

**Status: STUB — most steps require 2× p4d.24xlarge instances with EFA in a placement group.**

## Overview
Full distributed networking stack: EFA/RDMA transport, PTP clock sync,
gRPC control plane, AF_XDP kernel bypass, ring/tree/halving-doubling all-reduce,
Raft consensus, and chaos engineering harness.

## Steps

| # | Directory | What | Hardware |
|---|-----------|------|----------|
| 1 | efa_setup | EFA install, fi_pingpong baseline | 2× p4d in placement group |
| 2 | rdma_v1 | libfabric two-sided send/recv | same |
| 3 | rdma_onesided | fi_read/fi_write | same |
| 4 | efa_srd | SRD vs RC transport comparison | same |
| 5 | ptp | IEEE 1588 clock sync | same |
| 6 | grpc_control | gRPC + protobuf control plane | any Linux |
| 7 | flatbuffers_data | zero-copy serialization for data plane | any Linux |
| 8 | af_xdp | AF_XDP kernel bypass | any Linux |
| 9 | userspace_net | full userspace networking stack | any Linux |
| 10 | nic_deep_dive | NIC descriptor rings, RSS, PFC/ECN | any Linux |
| 11 | ring_allreduce | ring all-reduce from scratch | 2+ nodes |
| 12 | halving_doubling | recursive halving-doubling all-reduce | 2+ nodes |
| 13 | tree_allreduce | tree all-reduce | 2+ nodes |
| 14 | collectives | broadcast, reduce-scatter, all-gather | 2+ nodes |
| 15 | nccl_tuning | NCCL_ALGO/PROTO/BUFFSIZE tuning | GPU nodes |
| 16 | topo_scheduler | topology-aware scheduler | any Linux |
| 17 | vector_clocks | Lamport timestamps + vector clocks | any Linux |
| 18 | chandy_lamport | distributed global state capture | any Linux |
| 19 | raft | Raft consensus from scratch | any Linux |
| 20 | tla_raft | TLA+ spec + TLC model checker | any Linux |
| 21 | backpressure | token bucket + load shedding | any Linux |
| 22 | hedged_requests | hedged requests for tail latency | any Linux |
| 23 | multitenancy | resource quotas + fair scheduling | any Linux |
| 24 | chaos | tc netem injection harness | any Linux |
| 25 | tla_collective | TLA+ for all-reduce protocol | any Linux |

## Hardware notes
- EFA steps: 2× p4d.24xlarge in VPC placement group with EFA-enabled NICs
- Most steps (6+): any Linux instance (c5.2xlarge is fine)
