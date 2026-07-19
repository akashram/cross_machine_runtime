# NCCL Tuning

**Status: script complete, not yet run — requires GPU nodes with NCCL + nccl-tests.**

## What this measures
`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_BUFFSIZE` tuning for EFA topology, measured
collective throughput before/after tuning.

## Design
`sweep_nccl_config.sh` grid-searches 5 algorithms × 3 protocols × 3 buffer
sizes (45 configs) against `nccl-tests`' `all_reduce_perf`, plus one
baseline run with NCCL's own auto-tuner left on (`NCCL_ALGO`/`NCCL_PROTO`
unset) for comparison — the point of this step isn't just "find a fast
config," it's "does manual tuning beat NCCL's internal cost model, and by
how much." Output is CSV (`algo,proto,buffsize,busbw_gbs`) so it drops
straight into the Results table below. This step is a thin wrapper around
`this project's ring/tree/halving-doubling implementations (steps 11-13)
vs. NCCL` comparison already tracked in each of those steps' READMEs —
tuning NCCL itself matters because it's the baseline those comparisons are
measured against.

## Results
TODO: run on GPU nodes (p4d.24xlarge for real NVLink+EFA topology) once
`gpu_engine/` (Phase 3) hardware validation reaches this point.

| Config | Bus bandwidth (GB/s) | vs. auto-tuned baseline |
|--------|----------------------|--------------------------|
| Auto (NCCL default) | TODO | — |
| Best manual config | TODO | TODO |

## Hardware notes
- Required: GPU nodes with NCCL + nccl-tests built; EFA topology for the
  tuning to matter (single-node NVLink-only makes most of this moot)
