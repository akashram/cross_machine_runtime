# small_file_bottleneck

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 21 step 2: the real wall-clock and metadata-operation-count
(`open()` call count) cost of many-small-files vs. `data_loading/
webdataset_shard`'s existing shard-into-large-blobs approach, on this
Mac's local filesystem (APFS) — the real problem WebDataset-style
sharding exists to solve, demonstrated with this repo's own real
`tar_append`/`tar_finish`/`WebDatasetShardReader` codec, not a
reimplementation.

Disclosed honestly: this is a LOCAL-FILESYSTEM stand-in for a real
parallel-filesystem metadata server (VAST DNode/Lustre MDS/GPFS NSD), not
an equivalent measurement — a real metadata server serializes requests
over a network RPC path with its own contention behavior a local UNIX
filesystem doesn't reproduce. The `open()`-call-COUNT ratio (not the
absolute wall-clock, which is APFS-specific) is the mechanism-level
number that transfers to any metadata-server-backed filesystem.

## A real methodology bug, caught by running the test, fixed before capturing results

The first version of this benchmark's `read_many_small_files()` measured
file SIZE via `ifstream(..., ios::ate)` + `tellg()` — a metadata-only
stat, not an actual content read — while `read_one_shard()` fully parsed
and copied every sample's bytes. That's an apples-to-oranges comparison,
and it produced a real, wrong-looking result: under the debug (`-O0`)
preset, the single-shard path came out SLOWER for both write (an
unreserved `std::vector` growing via repeated `tar_append` reallocation)
and read (comparing full content decode against a stat-only call).
Fixed two things: `read_many_small_files()` now actually reads full file
contents (a fair comparison against `read_one_shard()`'s real decode),
and `write_one_shard()` now `reserve()`s its buffer up front (matching
what a real production shard writer would do, since the total size is
known ahead of time — removing an artifact of this test's own
construction, not a real cost webdataset sharding pays). Re-run under
`--preset release`, both directions flipped to the expected one (below).

## Results (captured 2026-09-16, Apple clang 14, `--preset release`, this Mac)

```
== Small-file bottleneck: 5000 files x 4096 bytes ==

  write: many-small-files = 863.28 ms | single-shard = 43.97 ms | speedup = 19.63x
  read:  many-small-files = 144.01 ms | single-shard = 12.32 ms | speedup = 11.69x
  open() call count: many-small-files = 10000 | single-shard = 2 | ratio = 5000x

PASS  both approaches transfer exactly the same total payload bytes -- the comparison isn't skewed by an unequal workload
PASS  writing one shard is faster than writing N separate small files on this local filesystem
PASS  reading one shard is faster than opening N separate small files on this local filesystem
PASS  the metadata-operation-count gap (open() calls) is orders of magnitude -- the mechanism-level effect that transfers to a real parallel filesystem's metadata server even though the absolute wall-clock numbers above are APFS-specific

PASS
```

## Findings

- **19.63x write speedup, 11.69x read speedup** for one WebDataset shard
  vs. 5000 separate small files, on real local disk I/O, once the
  methodology bug above was fixed — a real, measured demonstration of
  exactly the problem `webdataset_shard.h` exists to solve.
- **The `open()`-call-count gap (5000x) is far larger than the wall-clock
  gap (~12-20x)** — the mechanism-level number that transfers to a real
  parallel filesystem's metadata server even on filesystems whose
  per-open latency differs wildly from APFS's (a metadata-server-backed
  parallel FS typically pays a network round trip per `open()`, making
  the wall-clock gap potentially far LARGER there than the ~12-20x
  measured here).
- **A real debug-vs-release-build sensitivity, disclosed rather than
  hidden**: under `-O0` (debug preset), unreserved `std::vector` growth
  and metadata-only reads produced a misleading result in the OPPOSITE
  direction. This is a genuine methodological lesson (worth stating
  plainly): a benchmark's own implementation choices can dominate the
  effect being measured if not checked carefully, and building under
  `--preset release` (as this repo's other perf-sensitive benchmarks do)
  matters for wall-clock claims.

## Hardware notes

None — pure CPU, local filesystem (APFS on this Mac). See
`hpc_storage/DESIGN.md` for the local-filesystem-as-parallel-FS-stand-in
disclosure.
