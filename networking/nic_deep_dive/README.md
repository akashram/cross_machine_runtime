# NIC Hardware Deep Dive

**Status: script complete, not yet run — requires Linux.**

## What this measures
TX/RX descriptor ring structure, RSS configuration, PFC + ECN setup for
lossless fabric (DCQCN), hardware timestamping configuration.

## Design
`nic_dump.sh` is read-only diagnostics, not a tuning script — six sections
matching PLAN.md's step 10 bullet exactly:

1. **Descriptor rings** (`ethtool -g`) — TX/RX ring depth; too shallow
   under bursty load means drops before the CPU/AF_XDP poll loop ever
   sees a packet, which would confound `af_xdp`/`userspace_net`'s
   (steps 8-9) latency numbers if left unexamined.
2. **RSS** (`ethtool -l/-x/-n`) — queue count and hash indirection table;
   this determines which CPU core a given flow's interrupts/AF_XDP queue
   land on, which is why `nic_dump.sh` also dumps IRQ affinity (section 6)
   — RSS queue assignment and IRQ affinity have to agree for
   `cpu_engine`'s CPU-pinning work (Phase 2 step 1) to actually keep a
   flow's processing on one core.
3. **PFC** (`ethtool -a`, `dcb pfc show`) — per-priority pause frames,
   part of the lossless-fabric story for RDMA (EFA/RoCE-style fabrics
   assume no drops under congestion).
4. **ECN** (`tc -s qdisc`) — DCQCN's other half: congestion *signaling*
   without dropping, complementing PFC's pause-based backpressure.
5. **Hardware timestamping** (`ethtool -T`) — what `ptp/`'s PHC access
   (step 5) depends on; this section is the "does this NIC even support
   it" check that step assumes already passed.
6. **IRQ affinity** — cross-referenced against RSS queue count above.

## Results
TODO: run on Linux (ENA NIC on AWS) and record the actual output per
section.

## Hardware notes
- Required: Linux, `ethtool`, `iproute2` (`tc`, and `dcb` if available)
