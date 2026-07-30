# Custom schedulers vs. Kubernetes, compared directly

Phase 16 step 7. `networking/topo_scheduler` and `networking/multitenancy`
(Phase 5 steps 16 and 23, both code-complete and locally run — see their
own READMEs for exact measured behavior, summarized below) were built
from scratch. This is the honest, specific comparison PLAN.md asks for:
when is that still justified vs. reinventing Kubernetes' own scheduler
plus Kueue/Volcano. The two components get DIFFERENT verdicts — treating
them as one undifferentiated "our schedulers vs. theirs" question would
hide the actual finding.

## `topo_scheduler`: justified, and for a specific, narrow reason

**What it actually does** (`networking/topo_scheduler/README.md`):
given a `TopologyGraph` (measured or discovered PCIe/NVLink/EFA/FPGA
bandwidth between SPECIFIC device pairs — e.g. its own test models an
8-GPU node as two 4-GPU NVSwitch groups at 300 GB/s intra-group bridged
by one 25 GB/s cross-group link) and a workload's pairwise
communication-volume matrix, it computes all-pairs bandwidth-weighted
shortest paths (Floyd-Warshall) and greedily places the highest-volume
task pairs to minimize total weighted communication cost. Its own test
verifies exactly this: four high-volume "tensor-parallel duo" pairs each
land entirely within one NVSwitch group rather than split across the
slow bridge.

**What Kubernetes' native scheduler and Kueue/Volcano's topology
extensions actually reason about**: node/rack/zone-level labels and
(via kubelet's TopologyManager) NUMA-node alignment of CPU/memory/device
resources for a SINGLE pod's own allocation. Kueue's Topology Aware
Scheduling (TAS) and Volcano's binpack/topology plugins extend this to
gang-scheduled groups of pods, but still operate on a declared hierarchy
of labels (e.g. "same rack," "same zone") — neither ingests a real,
continuous bandwidth NUMBER between two specific accelerators the way
`topo_scheduler`'s `TopologyGraph` does, and neither optimizes a
whole-workload pairwise-communication-volume placement problem the way
`place()` does. This is the load-bearing difference, not "ours is more
educational": `topo_scheduler` answers a question — given THESE
measured bandwidths and THIS workload's actual communication pattern,
which specific placement minimizes total weighted transfer cost — that
Kubernetes' label-hierarchy model structurally cannot express, because
it has no notion of "25 GB/s between this exact device pair" as a first-
class scheduling input.

**Verdict**: justified, specifically for the fine-grained,
measured-bandwidth-driven placement decision within a single multi-
accelerator node or tightly-coupled node group. NOT a reason to
reimplement pod restart-on-crash, rolling updates, RBAC, or any of
Kubernetes' other machinery — `topo_scheduler`'s own scope is exactly
the placement DECISION, not a competing cluster orchestrator, and it
should hand off to Kubernetes (or Kueue/Volcano, expressed as node
affinity/anti-affinity hints derived from `place()`'s output) for
everything else.

## `multitenancy`: NOT justified as a cluster-level scheduler — Kueue/Volcano already do this, and do more

**What it actually does** (`networking/multitenancy/README.md`):
`FairScheduler` implements strict cross-class priority (every task from
a higher-priority tenant class runs before any lower-priority task
starts) plus weighted round-robin within a class, plus a fixed
per-tenant submission quota that rejects (not queues) once exceeded. Its
own measured test: a 2:1 configured weight ratio between two contending
tenants produces an exact 2.00 realized completion ratio, and a
20-of-50 quota is enforced exactly (20 accepted, 30 rejected).

**What Kubernetes already provides, natively or via Kueue/Volcano, for
the identical problem**: `PriorityClass` (built into the default
scheduler since 1.14) gives strict priority + preemption at the pod
level; `ResourceQuota` gives per-namespace resource caps (the
`multitenancy`'s fixed-quota-then-reject behavior, but scoped to actual
CPU/memory/GPU resource units, not an opaque task count); Kueue's
`ClusterQueue`/`LocalQueue` model gives weighted fair-sharing ACROSS
tenants with quota borrowing when a tenant is idle — a capability
`FairScheduler`'s fixed weighted round-robin does NOT have (its 2:1
split holds only while both tenants are actively contending; it has no
lending/borrowing concept for an idle tenant's unused share, unlike
Kueue's `borrowingLimit`/`lendingLimit`). On every axis `multitenancy`
implements, an existing, more capable, production-hardened Kubernetes
mechanism already covers the same ground and then some.

**Verdict**: not justified as a REPLACEMENT for Kubernetes-level
multi-tenant scheduling — Kueue/Volcano plus `PriorityClass`/
`ResourceQuota` should be used there, full stop. `FairScheduler`'s real,
narrower justification is operating BELOW the granularity Kubernetes
schedules at all: fair-sharing and priority admission for requests
WITHIN a single already-running process (e.g. `inference_serving`'s
`ServingRouter`, step 4's Deployment, fair-queuing individual inference
requests across API keys/tenants inside one pod) is a real, different
problem Kubernetes has no visibility into, because Kubernetes schedules
pods, not in-process request queues. That is the honest scope
`multitenancy` fits — not "a from-scratch alternative to Kueue," but "a
library usable INSIDE a pod, at a granularity Kubernetes doesn't reach."

## The general shape of the answer

Both verdicts follow the same underlying rule, worth stating once rather
than per-component: a from-scratch scheduler earns its keep exactly where
it operates on information or at a granularity Kubernetes' scheduling
model doesn't reach at all (measured device-pair bandwidth for
`topo_scheduler`; in-process request-level fairness for `multitenancy`),
and loses to Kubernetes/Kueue/Volcano everywhere the problem is actually
"schedule pods onto nodes fairly, with quotas and priority" — a solved,
mature problem this repo shouldn't re-solve just because Phase 5 built a
version of it to learn the mechanics.
