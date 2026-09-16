# minio_tuning

**Status: portable load-generator is code-complete AND locally run;
`deploy_minio.sh`/`tune_and_measure.sh` are real, complete, but
TOOLCHAIN-GATED — MinIO is not installed locally.**

## What this covers

PLAN.md Phase 21 step 11: the one step in this phase that involves
actually turning real knobs on a real (if not VAST-branded) storage
system, since no real VAST access exists (see `../vast_access_plan/
README.md`). MinIO was chosen specifically because it is
`brew`-installable with no Docker/cluster requirement — the realistic
default on this Mac, per PLAN.md's own framing of this step.

## Why this stayed unrun

This session, the user was offered four new local installs (PennyLane,
Strawberry Fields, MPI+OpenMP, and MinIO) and explicitly declined all
four: "none of these right now, just write the code and note these need
to be installed eventually." Per the standing no-new-local-installs
policy, `deploy_minio.sh` and `tune_and_measure.sh` are real, complete
MinIO commands — a 4-drive erasure-coded server startup, `mc` alias/
bucket setup, a real server-config tuning knob
(`api_requests_max`/`api_requests_deadline`), and a two-knob client-side
sweep (multipart `--part-size`, concurrent-connection parallelism via
parallel `mc cp` invocations) — written and ready to run, exactly the
same "real code, unrun, clearly marked TODO" convention as
`gpu_engine`'s CUDA code or `fpga_engine`'s HLS/TCL. Install (when
granted): `brew install minio/stable/minio minio/stable/mc`.

## What DOES run today: `minio_load_gen`

The portable half of this step — real, compiled, and run: generates
AI-workload-shaped I/O artifacts (WebDataset shards via the real
`tar_append`/`tar_finish` codec, checkpoint shards via the real
`write_shard_sync`) on local disk, ready to be fed into the MinIO scripts
above once MinIO is installed.

### Results (captured 2026-09-16, Apple clang 14, `--preset release`, this Mac)

```
Generating AI-workload-shaped I/O artifacts in /tmp/hpc_storage_minio_load
Generated 179.2 MB across webdataset/ and checkpoints/ -- ready for deploy_minio.sh + tune_and_measure.sh once MinIO is installed.
```

## Real knobs the (unrun) scripts actually turn

1. **Multipart part/chunk size** (`mc cp --part-size`, swept 16MiB/64MiB/
   128MiB) — trades per-part HTTP overhead against retry-on-failure
   granularity, directly relevant to the large sequential checkpoint
   writes `io_pattern_characterization` (step 1) measured.
2. **Concurrent-connection limit** (parallel `mc cp` invocations via
   `xargs -P`, swept 1/4/16) — relevant to the many-small-object
   webdataset shard set `small_file_bottleneck` (step 2) showed is
   metadata-operation-count-heavy.
3. **Erasure-coding parity level** (config-time, via
   `MINIO_STORAGE_CLASS_STANDARD` before `minio server` starts) — a
   third real, documented knob, not swept in the same run since it
   requires a server restart per setting.

## TODO: run on MinIO (once installed)

- `deploy_minio.sh <data_root>` — start the 4-drive erasure-coded server.
- `tune_and_measure.sh <load_dir> <results_file>` — sweep the two
  client-side knobs above against `minio_load_gen`'s real output,
  capturing real before/after timing.
- Fill in this README's results with the real measured numbers per knob
  setting, replacing this TODO section.

## Hardware/access notes

MinIO is a real, differently-branded S3-compatible object store, not a
VAST/WekaFS/Lustre-GPFS substitute — see `../vast_access_plan/README.md`
for what specifically this step does NOT close (VAST's DASE-specific
CNode/DNode balancing, similarity-based global reduction; Lustre's
MDS-contention behavior at real HPC-center scale).
