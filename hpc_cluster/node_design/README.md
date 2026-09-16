# node_design -- NUMA/PCIe/topology-aware HPC node design analysis

**Status: written analysis, composition only -- no new measurement. Every
claim below is grounded in an existing component's own documented
numbers, not invented for this step.**

## What this is

PLAN.md Phase 22 step 7: a written analysis connecting
`foundation/numa`, `fpga_engine/pcie_latency`, and
`networking/topo_scheduler`'s existing, previously-separate findings into
an explicit "how would I spec an HPC node" argument -- composition, not
new measurement, same convention as Phase 16 steps 6-7's device-plugin/
scheduler analyses.

## The three layers this composes

1. **Within a node, across NUMA domains** (`foundation/numa/numa.h`):
   memory access cost depends on which socket's DRAM a thread reads,
   not just whether the data is "in RAM." The header's own documented
   indicative figures (2-socket Intel Xeon): local DRAM alloc ~10-15ns
   vs. remote ~18-25ns (+50-100%), local cache miss ~80ns vs. remote
   ~140ns (+75%). `NumaArena::alloc_on_node`/`bind_thread_to_node`
   (real, Linux-`mbind`-backed, degrading to a single-node no-op on
   macOS) are the mechanism; the node-design implication is that a
   multi-socket node's effective memory bandwidth for a NUMA-naive
   workload is meaningfully below its nameplate aggregate bandwidth,
   and pinning threads to the socket that owns their working set (the
   exact mechanism this repo already built) is the fix, not more DRAM.

2. **Between a node's CPU and its accelerators, over PCIe**
   (`fpga_engine/pcie_latency/pcie_latency.cpp`): a single kernel
   invocation bundles at least four physically distinct costs -- BAR
   write (doorbell, pure PCIe MMIO write latency, no DMA), DMA
   descriptor processing + data transfer (measured via OLS regression
   across 5 buffer sizes, 4KB-64MB, separating a fixed per-transfer
   intercept from a size-dependent bandwidth slope), interrupt dispatch
   overhead, and poll-dispatch overhead at a fixed interval. This
   step's own numbers are still `TODO: run on F1` (no FPGA/PCIe device
   under test locally), but the DECOMPOSITION itself is the load-bearing
   node-design lesson: a PCIe-attached accelerator's latency floor for
   SMALL, LATENCY-SENSITIVE work (a single kernel launch, a small
   parameter update) is dominated by the fixed costs (doorbell +
   descriptor overhead + interrupt/poll dispatch), not the bandwidth
   slope -- so a node design optimizing purely for PCIe generation/lane
   count (bandwidth) without also budgeting for dispatch latency will
   under-serve small-message/high-frequency accelerator traffic
   (exactly the profile `inference_serving/flash_decoding`'s per-token
   decode-step kernel launches have).

3. **Between nodes, over the interconnect fabric**
   (`networking/topo_scheduler/topo_scheduler.cpp`): topology-aware
   placement matters at a coarser grain than either of the above --
   `topo_scheduler`'s real, locally-run sanity test models an 8-GPU node
   as two 4-GPU NVSwitch groups (300 GB/s intra-group) bridged by one
   slow cross-group link (25 GB/s, a 12x bandwidth cliff) and verifies
   its Floyd-Warshall-based placement algorithm keeps every high-volume
   "tensor-parallel duo" pair inside one NVSwitch group rather than
   split across the slow bridge. The node-design implication: a
   multi-GPU node's INTERNAL topology (which GPUs share an NVSwitch
   domain vs. which are bridged by a slower link) is exactly as
   consequential to real workload placement as the CROSS-node network
   topology `topo_scheduler` was originally built to reason about -- the
   same algorithm applies at both scales, which is why this repo built
   it generically rather than as two separate tools.

## The composed argument: three tiers, three orders of magnitude, one algorithm

Putting the three layers on one scale (using each component's own
documented/measured numbers, not new ones): NUMA cross-socket penalties
are a ~1.5-2x effect at ~10-100ns latencies; PCIe accelerator dispatch
overhead is a fixed-cost floor independent of transfer size, at
microsecond-to-tens-of-microsecond latencies (the exact quantity
`pcie_latency.cpp` is built to isolate, still `TODO` pending real F1
hardware); cross-node fabric topology is a ~10x+ bandwidth cliff
(`topo_scheduler`'s modeled 300 vs. 25 GB/s) at the coarsest grain. **A
real HPC node spec has to budget for all three simultaneously, because a
workload that gets NUMA placement and cross-node topology right but
ignores PCIe dispatch overhead for small, frequent accelerator calls
(or vice versa) still bottlenecks** -- exactly the situation
`inference_serving/flash_decoding`'s per-token kernel launches would hit
on a NUMA-naive, PCIe-dispatch-oblivious node even with a perfectly
topology-aware cross-node placement.

Concretely, spec'ing a training node for this repo's own real workloads
(`distributed_training/training_worker`'s real 4-rank process-per-rank
runs, `inference_serving/flash_decoding`'s real per-token decode kernel
launches once run on real GPU hardware) means:
- **Pin each rank's process to a single NUMA node** (the exact mechanism
  `NumaArena`/`bind_thread_to_node` already implement) rather than
  letting the OS scheduler migrate a rank's threads across sockets mid-run.
- **Group GPUs assigned to one tensor-parallel/pipeline-parallel group
  within one NVSwitch domain**, using `topo_scheduler::place()`'s exact
  algorithm, not a round-robin GPU assignment that could split a
  high-communication-volume pair across the slow inter-domain bridge.
- **Budget PCIe lane count for both bandwidth AND dispatch-latency
  headroom**: a node spec that only checks "enough aggregate GB/s" and
  ignores the fixed per-kernel-launch overhead `pcie_latency.cpp`
  isolates will under-serve latency-sensitive small-kernel workloads
  even with generous bandwidth.

## What this step doesn't (and can't yet) claim

The PCIe-layer numbers this analysis leans on are still `TODO: run on
F1` -- no FPGA/PCIe device exists locally to measure the real BAR-write/
descriptor-overhead/interrupt-dispatch/poll-dispatch split
`pcie_latency.cpp` is built to produce. This analysis's PCIe-layer
argument is therefore about the SHAPE of the cost (a fixed-cost floor
distinct from the bandwidth-dependent term) rather than specific numbers
-- exactly what the OLS-regression design of `pcie_latency.cpp` is built
to separate once it runs, and exactly the gap this analysis flags rather
than papers over.

## Hardware notes
None for this step itself (pure composition/writing). The components it
composes have their own hardware notes: `foundation/numa` real (Linux
`mbind`) vs. degraded (macOS, single-node) behavior is already validated
locally; `fpga_engine/pcie_latency` needs real F1 hardware (unrun);
`networking/topo_scheduler`'s placement algorithm is validated locally
via its own synthetic 8-GPU topology test, real topology discovery needs
a multi-GPU Linux node.
