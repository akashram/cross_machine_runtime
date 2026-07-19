# AF_XDP Kernel Bypass

**Status: code-complete, not yet built — requires Linux + libbpf + a NIC/driver with XDP support.**

## What this measures
XDP socket setup, UMEM region, TX/RX rings, measure latency vs. standard socket.

## Design
`XdpSocket` (`af_xdp_socket.cpp`) follows libbpf's `xdpsock` reference tool
structure: one mmap'd UMEM region sliced into fixed-size frames, four
rings (fill/completion for RX-side/TX-side frame handoff to the kernel,
RX/TX for actual packet descriptors). Setup order matters and is easy to
get backwards silently: the UMEM must exist and its fill ring must be
pre-populated with every frame index **before** `xsk_socket__create` — the
kernel starts trying to use the fill ring the instant the socket exists,
so creating it first means RX silently never delivers anything (no error,
just an empty ring forever — documented in the file-level comment as the
"classic AF_XDP bug" specifically because it doesn't fail loudly).
`transmit()` reclaims frames from the completion ring before allocating a
new one — TX frames are a finite pool, not allocated fresh per packet.

## Results
TODO: run on Linux with XDP-capable NIC (native XDP requires driver
support — check with `ip link show` after loading; falls back to generic
/ SKB mode otherwise, much slower).

| Mode | Latency (µs) | Throughput (Mpps) | CPU util |
|------|-------------|--------------------|----------|
| Standard socket (baseline) | TODO | TODO | TODO |
| AF_XDP generic mode | TODO | TODO | TODO |
| AF_XDP native mode | TODO | TODO | TODO |

## Hardware notes
- Required: Linux, libbpf with `bpf/xsk.h`, NIC/driver with XDP support
  (ENA on AWS supports XDP in at least generic mode)
