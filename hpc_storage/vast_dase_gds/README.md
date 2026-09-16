# vast_dase_gds

**Status: literature/vendor-doc-grounded written analysis. No code — see
"Definition of done" below for why.**

## What this covers

PLAN.md Phase 21 step 4: a detailed treatment of VAST Data's Disaggregated
Shared Everything (DASE) architecture, and how it plugs into this repo's
existing `distributed_training/gpudirect_storage` cuFile client as a real
GDS-certified backend. No VAST hardware or vendor POC access exists
locally (see PLAN.md's Phase 21 hardware-access note and
`vast_access_plan/README.md`), so this is grounded in VAST's published
architecture whitepapers and public GDS certification listings, not a
measurement — the same honest-labeling convention `storage_comparison`
(step 3) and `analog_engine/nvm_comparison` already use.

## DASE architecture, in detail

VAST's core architectural claim is full disaggregation of two roles that
traditional parallel filesystems (Lustre's OSS/OST, GPFS's NSD) couple
more tightly:

- **CNodes** (Compute Nodes): stateless, handle protocol termination
  (NFS/SMB/S3/GDS) and data-reduction/erasure-coding logic. Stateless
  means CNodes scale independently of capacity — adding protocol
  throughput doesn't require adding storage media.
- **DNodes** (Data Nodes): hold the actual media — QLC flash for bulk
  capacity (cheaper per-GB than TLC/SLC, but slower write endurance,
  which is exactly why the next point matters) plus a small pool of
  Storage Class Memory / NVRAM as a write buffer that absorbs bursty
  writes before flushing to QLC in patterns that don't shorten QLC's
  write endurance.

This is the structural difference from Lustre/GPFS `storage_comparison`
(step 3) already flagged as the small-file/metadata bottleneck: Lustre's
MDS (metadata server) is a specific, separately-scaled tier that can
become a contention point under AI training's shuffled small-file read
pattern. VAST's **global namespace** spreads metadata across the whole
DNode pool rather than concentrating it in one server role, and
**similarity-based data reduction** (VAST's own "Global Reduce") operates
across the ENTIRE namespace rather than per-node/per-volume — the
practical consequence (per VAST's published material) is that reduction
ratios don't degrade as cluster size grows, unlike per-node dedup schemes
whose effective window is bounded by what fits on one node.

## Connection to `distributed_training/gpudirect_storage`

`gpudirect_storage/gds_reader.h` (this repo's real, hardware-gated cuFile
client — see that step's own README) is backend-agnostic: `GdsFileReader`
opens a file with `O_DIRECT` and registers it with the cuFile driver via
`cuFileHandleRegister`, which works against ANY GDS-certified storage
backend the underlying filesystem mount presents as a POSIX path — VAST
is not special-cased in that code, and should not be. What VAST's
architecture changes is what happens BEHIND that mount point once a read
is issued:

1. The GPU issues a cuFile read against a VAST-mounted path.
2. The request lands on a CNode, which (being stateless) can be any CNode
   in the cluster — there's no "home node" affinity the way Lustre's
   OST-ownership model creates.
3. The CNode fetches the (deduplicated, erasure-coded) data from across
   the DNode pool and streams it back via NVMe-oF/RDMA, still bypassing
   host CPU staging — GDS's whole value proposition
   (`gds_reader.h`'s own header comment already documents this bypass;
   VAST's contribution is what the DMA path's OTHER end looks like).

The honest, disclosed limitation: this repo's `gds_reader.h` is unrun (no
GDS-capable NVMe device or CUDA toolchain locally — see that step's own
README), so there is no real measurement anywhere in this repo of
GDS-over-VAST specifically vs. GDS-over-a-simpler-local-NVMe-target. This
step's contribution is the architectural connection and its rationale,
not a new measurement — consistent with PLAN.md's own framing of step 4
as a "deep dive... extends that step's existing hardware-gated context
rather than replacing it."

## Sources

VAST Data's published DASE architecture whitepaper and NVIDIA's GPUDirect
Storage certified-partner documentation — see `READING_LIST.md`'s Phase 21
section for full citations.

## Definition of done

Per PLAN.md's Phase 21 "Definition of done": steps that can't produce a
real local measurement (3, 4, 5, 8, 10 — this step among them) are
literature/vendor-doc-grounded and explicitly labeled as such rather than
presented as measured. No `.cpp`/`.h` exists for this step because there
is nothing here that isn't either (a) already real code elsewhere in this
repo (`gpudirect_storage/gds_reader.h`, referenced not duplicated) or
(b) a written architectural claim with no local mechanism to run.
