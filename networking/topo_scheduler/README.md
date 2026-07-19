# Topology-Aware Scheduler

**Status: code-complete AND locally run — portable placement algorithm, real discovery deferred to GPU/Linux hardware.**

## What this measures
PCIe tree discovery, NVLink topology (nvml), EFA bandwidth map, FPGA
attachment paths. Scheduler uses this map for placement decisions.

## Design
Split into two concerns this project keeps separate: **discovery**
(PCIe/NVLink/EFA/FPGA topology — genuinely Linux/hardware-specific, not
implemented here) populates a `TopologyGraph` (`topo_scheduler.h`); the
**placement algorithm** (`topo_scheduler.cpp`) — given that graph and a
workload's pairwise communication volumes — decides which task runs on
which node, and has nothing hardware-specific about it. `place()`
precomputes all-pairs bandwidth-weighted shortest paths (Floyd-Warshall,
cost = sum of 1/bandwidth along a path — chained hops add, like series
resistance), then greedily places task pairs in descending
communication-volume order, each unplaced task going to whichever free
node minimizes cost to its already-placed partners. Greedy, not globally
optimal (graph partitioning is NP-hard) — same documented stance as
`compiler/placement`'s device-placement pass (Phase 4 step 9): validate
the mechanic now, revisit the algorithm once a real workload's numbers
justify a fancier one.

## Sanity-run output (Mac, 2026-07-19)

`topo_scheduler_test` models an 8-GPU node: two 4-GPU NVSwitch groups
(300 GB/s intra-group) bridged by one slow cross-group link (25 GB/s).
Workload: four high-volume "tensor-parallel duo" pairs plus three
low-volume "pipeline-adjacent" pairs linking the duos. A correct
topology-aware placement keeps every high-volume duo within one NVSwitch
group rather than split across the slow bridge:

```
duo 0 (tasks 0,1) -> nodes gpuA0,gpuA1: same NVSwitch group
duo 1 (tasks 2,3) -> nodes gpuA2,gpuA3: same NVSwitch group
duo 2 (tasks 4,5) -> nodes gpuB0,gpuB1: same NVSwitch group
duo 3 (tasks 6,7) -> nodes gpuB2,gpuB3: same NVSwitch group
PASS
```

## Results
TODO: run on real multi-GPU hardware — populate `TopologyGraph` from
actual `nvidia-smi topo -m` / nvml output, and compare against a
topology-*unaware* (e.g. round-robin) placement to quantify the win.

| Placement strategy | Cross-slow-link bytes moved | Effective collective bandwidth |
|---------------------|------------------------------|----------------------------------|
| Round-robin (baseline) | TODO | TODO |
| Topology-aware (`place()`) | TODO | TODO |

## Hardware notes
- `TopoScheduler` builds and runs anywhere (validated on Mac). Real
  topology discovery needs a multi-GPU Linux node (nvml for NVLink, sysfs
  for PCIe tree).
