# pingpong — double-buffered compute/transfer overlap

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize.**

## What this measures
A 64-block, 1024-element-per-block transform, implemented two ways in
`pingpong.cpp`:

- `pingpong_single_buffered`: one on-chip buffer, reused every block.
  Load, compute, and store for block N must fully serialize before block
  N+1's load can start — no overlap is possible regardless of scheduling,
  since there's only one copy of the storage to write into.
- `pingpong_double_buffered`: two on-chip buffers (`buf[2][kBlockSize]`,
  `ARRAY_PARTITION dim=1 complete` so they're independently addressable),
  block index selects the active one. `DATAFLOW` at the block loop lets
  Vitis HLS overlap block N+1's `load_block` (writing into the *other*
  buffer) with block N's `compute_store_block` (reading the current one) —
  the throughput win this step exists to quantify, not assume.

This is a different overlap axis than `loop_opt_dataflow`: that kernel
streams single elements continuously through 3 stages; this one stages
whole blocks into BRAM (because the stand-in compute here needs the full
block resident, not just one element at a time) and overlaps at block
granularity instead.

## Results
TODO: synthesize on F1.

| Variant | Total latency (cycles) | Throughput (elements/cycle) | BRAM |
|---|----|----|----|
| `pingpong_single_buffered` | TODO | TODO | TODO |
| `pingpong_double_buffered` | TODO | TODO (target: ~block load/compute overlap) | TODO |

## Hardware notes
- Required: AWS F1, Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl`
