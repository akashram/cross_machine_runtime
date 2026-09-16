# vast_access_plan

**Status: written plan. No code — this step's entire deliverable is
honesty about an access gap, not a measurement.**

## What this covers

PLAN.md Phase 21 step 10: documenting, honestly, that VAST Data / WekaFS /
Lustre-GPFS access does NOT follow the "spin up a spot instance" pattern
this repo's hardware-validation table uses for GPU/FPGA/TPU/QPU (Phase 20).

## Why this is a genuinely different kind of gap

Every other hardware-gated phase in this repo (3, 7, 8, 15, 20) has a
straightforward, if costly, path to real hardware: an hourly cloud rental
(g4dn.xlarge for GPU, F1 spot for FPGA, a TPU v4-8 VM, IBM Quantum's free
tier or AWS Braket for QPU). None of VAST Data, WekaFS, or a real
production-scale Lustre/GPFS deployment has an equivalent hourly-rental
story:

- **VAST Data** is sold as an enterprise appliance / managed deployment.
  Real access is typically a vendor proof-of-concept (POC) engagement —
  VAST provisions a demo cluster (often remotely accessible) for a scoped
  evaluation period, arranged directly with VAST or a reseller, not a
  self-service cloud console.
- **WekaFS** has a more cloud-native deployment story (it runs on
  standard cloud NVMe instances) and could in principle be
  self-provisioned on AWS/GCP compute — but this still means paying for
  the underlying compute AND a Weka software license/trial, not a simple
  per-hour managed-service rate, and no free/trial tier exists at the
  scale needed to demonstrate real multi-node behavior.
- **Lustre/GPFS** at real HPC-center scale is typically only accessible
  via an existing HPC center allocation (a university/national-lab
  cluster) or a from-scratch self-managed deployment (BeeGFS/Lustre
  community edition on rented compute + block storage) — the latter is
  possible but produces a DIY approximation, not the tuned,
  production-representative deployment a real HPC storage engineering
  role would actually work with.

## What a real validation pass would need, per system

| System | Access path | What it would validate |
|---|---|---|
| VAST Data | Vendor POC/demo engagement (contact VAST or a reseller) | Real DASE small-file/metadata numbers vs. `small_file_bottleneck`'s local-FS proxy; real GDS-over-VAST throughput vs. `gpudirect_storage`'s unrun cuFile client |
| WekaFS | Self-provision on cloud NVMe instances + Weka trial license | Same small-file/GDS comparison, software-defined deployment model |
| Lustre/GPFS | HPC center allocation, or self-managed Lustre/BeeGFS on rented compute+block storage | Real MDS-contention behavior under a shuffled small-file AI training read pattern — the specific bottleneck `storage_comparison` (step 3) predicts from documentation, not measurement |
| Ceph | Fully self-provisionable (no vendor gate) — closest to step 11's MinIO in access model, just heavier | Real CRUSH-map/placement-group tuning at multi-node scale, beyond step 11's single-node MinIO substitute |

## Relationship to step 11 (MinIO hands-on tuning)

`minio_tuning/` (step 11) does NOT close this gap — see that step's
README for the explicit disclosure. MinIO is a real, differently-branded
S3-compatible object store, chosen specifically because it is
`brew`-installable with no Docker/cluster requirement, giving genuine
hands-on tuning experience (erasure coding, part-size, concurrency
knobs) on SOME real system. It does not exercise VAST's DASE-specific
surface (CNode/DNode balancing, similarity-based global reduction) or
Lustre/GPFS's metadata-server contention behavior — those stay genuinely
gated behind the access paths in the table above.

## Definition of done

This step's deliverable IS the honest documentation above — there is no
code to write, and pretending otherwise (e.g. a fake "VAST simulator")
would misrepresent what's actually validated vs. assumed, the opposite of
this repo's disclosure convention.
