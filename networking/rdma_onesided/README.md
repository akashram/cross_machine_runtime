# RDMA One-Sided Operations (Read/Write)

**Status: STUB — requires EFA-enabled instances.**

## What this measures
fi_read/fi_write (RDMA read/write without remote CPU involvement). Compares
latency and bandwidth vs two-sided send/recv.

## Results
TODO: run on EFA hardware.

| Op | Size | Latency (µs) | BW (GB/s) |
|----|------|-------------|-----------|
| fi_write (1KB) | 1 KB | TODO | TODO |
| fi_read (1KB) | 1 KB | TODO | TODO |
| fi_write (1MB) | 1 MB | TODO | TODO |

## Hardware notes
- Required: EFA-enabled instances with registered memory regions
