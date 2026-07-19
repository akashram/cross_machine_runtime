# RDMA One-Sided Operations (Read/Write)

**Status: code-complete, not yet built — requires EFA-enabled instances.**

## What this measures
fi_read/fi_write (RDMA read/write without remote CPU involvement). Compares
latency and bandwidth vs two-sided send/recv.

## Design
Free-function API over one implicit module-level endpoint
(`rdma_onesided.cpp`) rather than a class — a one-sided benchmark only
ever drives one connection. `rdma_register_memory` must run on **both**
peers before either calls `rdma_write`/`rdma_read`: the FI_REMOTE_READ |
FI_REMOTE_WRITE-flagged registration is what lets the *other* side's NIC
touch this memory without this process's CPU ever running a matching
recv() — that's the whole point of one-sided RDMA vs. `rdma_v1`'s
two-sided send/recv. Same fabric/domain/endpoint object graph as
`rdma_v1/RdmaEndpoint`, extended with `FI_RMA` capability and
`FI_READ|FI_WRITE` completion-queue binding.

## Results
TODO: run on EFA hardware.

| Op | Size | Latency (µs) | BW (GB/s) |
|----|------|-------------|-----------|
| fi_write (1KB) | 1 KB | TODO | TODO |
| fi_read (1KB) | 1 KB | TODO | TODO |
| fi_write (1MB) | 1 MB | TODO | TODO |

## Hardware notes
- Required: EFA-enabled instances with registered memory regions
