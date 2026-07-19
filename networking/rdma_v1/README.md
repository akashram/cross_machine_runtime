# RDMA Transport v1 (Two-Sided Send/Recv)

**Status: `rdma_transport` (libfabric/EFA) code-complete, not yet built —
requires EFA-enabled instances. `tcp_baseline_bench` code-complete AND
locally run — see below.**

## What this measures
libfabric two-sided send/recv over EFA. Measures latency (RTT) and throughput
vs TCP socket baseline.

## Design
`RdmaEndpoint` (`rdma_transport.cpp`) follows libfabric's `fi_pingpong`
reference structure: `fi_getinfo` resolves an EFA-capable RDM provider,
`fi_fabric`/`fi_domain`/`fi_endpoint` stand up the object graph, one shared
completion queue for send and recv, one address-vector entry per peer
(RDM endpoints are connectionless, so `listen()` is really "wait for a
peer's `connect()` to populate the AV," not a socket-style accept). Gated
behind Linux + libfabric in `CMakeLists.txt` — can't be built or exercised
without an EFA NIC.

`tcp_baseline_bench.cpp` is standalone (not built on
`networking/common::Channel` — see that component's README for why: a real
2-node deployment needs asymmetric bind/connect addresses Channel's
loopback-oriented mesh doesn't model) and measures real p50/p99 RTT +
bandwidth at the same 64B/4KB/1MB sizes `fi_pingpong` uses, so the two
numbers in the table below are directly comparable once the EFA column is
filled in.

## Sanity-run output (Mac, loopback, 2026-07-19)

`tcp_baseline_bench loopback` — real TCP over 127.0.0.1, 200 round trips
per size:

```
size             p50 (us)     p99 (us) bandwidth (GB/s)
64                  31.90       105.53          0.003
4096                29.91        46.10          0.258
1048576            384.32       700.35          3.565
```

Small-message latency is dominated by loopback syscall/scheduling overhead
here, not the network — real cross-node TCP numbers (and the EFA
comparison) require the actual 2-node setup below.

## Results
TODO: run on EFA hardware for the RDMA column; re-run `tcp_baseline_bench
client`/`server` across two real nodes for the TCP column (the loopback
numbers above are a sanity check, not the real baseline).

| Message size | RDMA RTT (µs) | TCP RTT (µs) | Speedup |
|--------------|--------------|-------------|---------|
| 64 B | TODO | TODO | TODO |
| 4 KB | TODO | TODO | TODO |
| 1 MB | TODO | TODO | TODO |

## Hardware notes
- Required: 2× EFA-enabled instances (p4d.24xlarge)
- `tcp_baseline_bench` builds and runs anywhere (validated on Mac); real
  2-node numbers just need `server`/`client <ip>` run across the same two
  instances used for the EFA column.
