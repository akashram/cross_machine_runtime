# Phase 21: HPC Storage Engineering for AI Workloads

**Status: CODE COMPLETE (11/11 steps), 2026-09-16.**

## Overview

Scoped 2026-09-12 while checking this repo against what an HPC storage
engineer role (e.g. tuning VAST Data for AI training workloads) actually
needs: real code already existed for the compute-side half of the storage
story (`distributed_training/gpudirect_storage`'s cuFile client, the
offload logic, checkpoint sharding), but nothing about the parallel
filesystem layer itself or the I/O-pattern/capacity-planning reasoning a
storage engineer does around it. This phase closes that gap.

Unlike every other hardware-gated phase in this repo, there is no simple
hourly cloud rental for VAST Data/WekaFS/Lustre-GPFS (see
`vast_access_plan/README.md`) — a genuinely different access path from
GPU/FPGA/TPU/QPU's "spin up a spot instance" pattern. Six of eleven steps
still produce a real local measurement (this repo's own data-loading/
checkpoint code, its own local filesystem, its own trained model
artifacts, and a real `brew`-installable object store); five stay
literature/vendor-doc-grounded, explicitly labeled as such. See
`hpc_storage/DESIGN.md` for the full phase-level design rationale.

No new local installs were made for this phase — the user explicitly
declined all four install offers this session ("none of these right now,
just write the code and note these need to be installed eventually"),
including MinIO for step 11. Step 11 is therefore real, complete,
toolchain-gated code (deployment scripts + tuning commands), unrun,
except for its portable load-generator half, which does run today.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | `io_pattern_characterization` | Real I/O pattern of `data_loading`/`checkpoint`, measured | Run locally |
| 2 | `small_file_bottleneck` | Many-small-files vs. sharded-blob metadata cost, measured | Run locally |
| 3 | `storage_comparison` | VAST/WekaFS/Lustre-GPFS/Ceph comparison | Literature-grounded |
| 4 | `vast_dase_gds` | VAST DASE architecture + GDS integration | Literature-grounded, no code |
| 5 | `nfs_rdma_tuning` | NFSoRDMA/RoCE tuning for bulk storage traffic | Literature-grounded, no code |
| 6 | `checkpoint_burst_capacity` | Real thundering-herd capacity model, measured | Run locally |
| 7 | `data_reduction` | Real compression ratios on this repo's real artifacts | Run locally |
| 8 | `storage_qos` | Storage bandwidth/IOPS QoS, extends `networking/multitenancy` | Run locally |
| 9 | `dlio_bench` | Real DLIO-style AI I/O benchmark | Run locally |
| 10 | `vast_access_plan` | VAST/WekaFS/Lustre-GPFS access-path plan | Written plan, no code |
| 11 | `minio_tuning` | Real hands-on MinIO tuning | Toolchain-gated (load-gen runs locally) |

## Design highlights — how the steps chain together

- **Step 1 grounds every later step** in this repo's own REAL I/O
  pattern (few large sequential writes on the checkpoint path, many
  small sequential reads on the load path) rather than an assumed
  profile — steps 2 and 6 build directly on that finding.
- **Step 2 reuses step 1's exact synthetic-dataset generation shape**
  and the real `tar_append`/`tar_finish`/`WebDatasetShardReader`
  functions `data_loading/webdataset_shard.h` already defines.
- **Step 6 reuses the real `AsyncCheckpointWriter`** class
  `distributed_training/checkpoint` already implements, run N-way
  concurrently to measure real contention on one shared disk — not a
  simulated queueing model.
- **Step 7 reuses the exact training recipe** `npu_engine/quant_export`
  already validated (same transformer/corpus shape), so its real trained
  `w_out` weight is a genuine post-training artifact, not a
  randomly-initialized stand-in.
- **Step 8 composes the real, unmodified `networking::multitenancy::
  FairScheduler`** rather than reimplementing priority/quota admission —
  only the storage-specific bandwidth token bucket is new code.
- **Step 9 composes step 1's real `DataLoader` usage with a real
  `transformer/` forward pass**, interleaved exactly as a training loop
  would, which is DLIO's actual differentiator from a generic filesystem
  benchmark.
- **Step 11's load generator reuses the same real codecs** steps 1/2/6
  exercise, so the (unrun) MinIO tuning scripts have real AI-workload-
  shaped data ready once MinIO is installed.
- **Steps 3/4/5/10 are honestly literature-grounded**, cross-referencing
  `distributed_training/gpudirect_storage` (step 4) and
  `networking/rdma_v1`/`nic_deep_dive` (step 5) rather than duplicating
  their own hardware-gated status.

See each step's own README for full methodology and captured output;
`hpc_storage/DESIGN.md` for the phase-level design rationale.

## Hardware/access notes

No VAST Data, WekaFS, or production-scale Lustre/GPFS deployment exists
locally, and unlike GPU/FPGA/TPU/QPU there is no simple hourly cloud
rental for any of them — see `vast_access_plan/README.md` for the full,
system-by-system access-path breakdown. Step 11's MinIO deployment does
NOT close this gap (see that step's own disclosed limitation).

## Next

Phases 20 (Quantum Computing) and 22 (HPC Cluster Systems Engineering)
were scoped alongside this phase and, per the user's explicit direction,
are being implemented in parallel this same session in separate git
worktrees — see `quantum_engine/README.md` and `hpc_cluster/README.md`
(once their own wrap-ups land) for progress.
