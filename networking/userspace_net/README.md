# Userspace Networking Stack

**Status: code-complete, not yet built — built on af_xdp (step 8), same Linux + libbpf requirement.**

## What this measures
Full userspace networking stack: AF_XDP (or DPDK) send/recv pipeline with measured latency.

## Design
`NetStack` (`net_stack.cpp`) is a thin message-framing layer over step 8's
`XdpSocket`: a 4-byte magic + 4-byte length header distinguishes this
traffic from anything else sharing the NIC in generic/SKB XDP mode, and
`send()`/the receive callback deal in payloads, not raw frames. The
receive loop is a **busy-poll**, deliberately — an `epoll`-based blocking
wait would reintroduce the syscall/interrupt latency AF_XDP exists to
avoid, defeating the point of this step. This is the same tradeoff
`cpu_engine`'s busy-poll-vs-OS-wait comparison (Phase 2 step 13) measured
for the SPSC ring buffer case: burn a full core, get lower and
more-predictable tail latency.

## Results
TODO: run on Linux with XDP-capable NIC.

| Metric | Standard socket | userspace_net (AF_XDP) |
|--------|-----------------|-------------------------|
| p50 latency (µs) | TODO | TODO |
| p99 latency (µs) | TODO | TODO |
| Throughput (Mpps) | TODO | TODO |
| CPU utilization | TODO | TODO (expect ~100% on the poll core) |

## Hardware notes
- Required: Linux, libbpf, XDP-capable NIC/driver (same as af_xdp/)
