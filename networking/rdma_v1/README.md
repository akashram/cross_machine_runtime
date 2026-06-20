# RDMA Transport v1 (Two-Sided Send/Recv)

**Status: STUB — requires EFA-enabled instances.**

## What this measures
libfabric two-sided send/recv over EFA. Measures latency (RTT) and throughput
vs TCP socket baseline.

## Results
TODO: run on EFA hardware.

| Message size | RDMA RTT (µs) | TCP RTT (µs) | Speedup |
|--------------|--------------|-------------|---------|
| 64 B | TODO | TODO | TODO |
| 4 KB | TODO | TODO | TODO |
| 1 MB | TODO | TODO | TODO |

## Hardware notes
- Required: 2× EFA-enabled instances (p4d.24xlarge)
