# EFA Setup + Baseline Latency

**Status: script complete, not yet run — requires 2× p4d.24xlarge in EFA placement group.**

## What this measures
AWS EFA installer validation, fi_info device enumeration, fi_pingpong latency
and bandwidth baseline against TCP socket baseline.

## Design
`efa_bench.sh verify` checks device enumeration (`fi_info -p efa`,
`ibv_devinfo`); `server`/`client` run `fi_pingpong` at 64B/4KB/1MB message
sizes (matching `rdma_v1/README.md`'s table so the two are directly
comparable). The TCP baseline column is **not** measured with netcat here —
it's `rdma_v1/tcp_baseline_bench`, a real compiled benchmark over
`networking/common/TcpChannel` at the same message sizes, so the comparison
is apples-to-apples (same framing/timing methodology, only the transport
differs). See `rdma_v1/README.md`.

## Results
TODO: run on EFA hardware.

| Metric | EFA | TCP baseline (`tcp_baseline_bench`) |
|--------|-----|-------------|
| Latency p50 (µs) | TODO | TODO |
| Latency p99 (µs) | TODO | TODO |
| Bandwidth (GB/s) | TODO | TODO |

## Hardware notes
- Required: 2× p4d.24xlarge, same VPC placement group, EFA-enabled
- Install: aws-efa-installer from AWS documentation
