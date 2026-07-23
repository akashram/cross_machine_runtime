# bram_uram — BRAM (dual-port) vs URAM (single-port) weight storage

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize.**

## What this measures
A 16384-entry, 16-bit weight table (256Kb) with an access pattern that
deliberately issues **two independent lookups per cycle**, so the
BRAM-vs-URAM port-count difference actually shows up in the schedule
instead of being hidden by a pattern only URAM can serve anyway:

- `bram_uram_bram`: `BIND_STORAGE ... type=RAM_2P impl=BRAM`. BRAM36
  blocks are true dual-port — expected to sustain both lookups in one
  cycle natively, at the cost of needing more BRAM36 blocks for the same
  capacity than a single URAM288 block would.
- `bram_uram_uram`: `BIND_STORAGE ... type=RAM_1P impl=URAM`. URAM288 is
  single-port — Vitis HLS has to either serialize the two lookups
  (doubling II for that loop) or the schedule changes some other way.
  Left as a genuinely single port on purpose: the II penalty this forces
  *is* the measurement, not something to design around.

The tradeoff this step exists to quantify: URAM's much larger capacity per
block (288Kb vs. 36Kb) against the II cost of its single port for this
specific two-reads-per-cycle pattern — i.e., whether the capacity win is
worth the throughput loss for weight-table-shaped access.

## Results
TODO: synthesize on F1.

| Variant | BRAM36 used | URAM288 used | II (lookup loop) | Fmax (MHz) |
|---|----|----|----|----|
| `bram_uram_bram` | TODO | 0 | TODO | TODO |
| `bram_uram_uram` | 0 | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1, Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl`
