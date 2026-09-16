# storage_comparison

**Status: code-complete AND locally run (the comparison table/scoring
logic itself runs and is tested); the underlying data is
literature/vendor-doc-grounded, not measured — see below.**

## What this covers

PLAN.md Phase 21 step 3: a parallel/distributed storage system comparison
(VAST Data, WekaFS, Lustre/GPFS, Ceph) for AI training workloads. No
rentable access to any of these four systems exists locally (see
`../vast_access_plan/README.md`), so every field in `storage_comparison.h`
is a literature/vendor-documentation-grounded representative
characterization — the same honest-labeling convention
`analog_engine/nvm_comparison` established for Phase 17's NVM device
comparison with no fab access.

## Results (captured 2026-09-16, Apple clang 14, `--preset release`, this Mac)

```
== Parallel/distributed storage comparison (literature/vendor-doc-grounded) ==

VAST Data    | Disaggregated Shared Everything (DASE): stateless CNodes + QLC-flash/NVRAM DNodes, global namespace | metadata=5/5 | GDS=yes | reduction=Global similarity-based reduction (cluster-wide, not per-node)
WekaFS       | Distributed POSIX filesystem over NVMe-oF, software-defined (commodity or appliance) | metadata=5/5 | GDS=yes | reduction=Per-cluster compression + data reduction
Lustre/GPFS  | Classic parallel FS: OSS/OST (Lustre) or NSD (GPFS) object/block servers + a metadata server tier | metadata=2/5 | GDS=no | reduction=Filesystem/hardware-level compression (not workload-aware global dedup)
Ceph         | Software-defined RADOS object store (CRUSH placement) underneath CephFS/RBD/RGW | metadata=3/5 | GDS=no | reduction=None built-in at the RADOS layer (relies on underlying block/OSD storage)

== Illustrative AI-workload-fit composite score ==
  VAST Data    1.000
  WekaFS       1.000
  Ceph         0.300
  Lustre/GPFS  0.200
PASS  four storage classes compared, as PLAN.md specifies
PASS  the two GDS-certified, high-metadata-rated systems (VAST, WekaFS) rank above the two that aren't/lower-rated (Lustre/GPFS, Ceph) under this illustrative scoring
PASS  Lustre/GPFS's centralized-metadata-server architecture (the well-documented real bottleneck for AI training's small-file pattern) scores lowest under this illustrative metric

PASS
```

## Findings

- VAST Data and WekaFS tie under this illustrative composite score
  (both GDS-certified, both rated 5/5 on small-file/metadata fitness) —
  consistent with `../vast_dase_gds/README.md`'s finding that the real
  differentiator between them is deployment model (fixed appliance vs.
  software-defined), not raw AI-workload-fitness ranking.
- Lustre/GPFS's classic centralized-metadata-server architecture scores
  lowest — the well-documented real bottleneck `small_file_bottleneck`
  (step 2) demonstrates the underlying MECHANISM of (many `open()` calls
  contending for one metadata path) on this repo's own local filesystem.
- Ceph sits in the middle: no hard disqualifier, but general-purpose by
  design rather than purpose-optimized for AI training's specific I/O
  shape the way VAST/WekaFS are — and no mainstream GDS certification as
  of this writing.

## Sources

See `hpc_storage/DESIGN.md` and `READING_LIST.md`'s Phase 21 section for
full citations (VAST DASE whitepapers, WekaFS architecture docs, Lustre/
GPFS's published HPC-deployment literature, Ceph/RADOS papers).

## Hardware notes

No VAST/WekaFS/Lustre/GPFS/Ceph deployment exists locally — every
numeric field is a literature-informed representative point, not a
measurement. See `../vast_access_plan/README.md` for the full access-path
breakdown.
