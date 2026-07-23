# slr — SLR partitioning and crossing penalty

**Status: hardware pblock placement is code complete and unrun (F1 required);
the crossing-penalty model is measured locally.**

## What this measures
Two things, deliberately kept separate since only one needs an F1 instance:

1. **Crossing penalty model** (`slr_crossing_model.cpp`): VU9P is a 3-SLR
   SSI device — SLR0/1/2 are separate dies joined by Laguna sites (SLLs),
   which add real routing delay any signal crossing an SLR boundary pays
   that a same-SLR signal doesn't. This model asks two questions locally:
   does a given kernel's logic even need to span SLRs by resource count
   alone, and if a crossing lands on an already-tight critical path (reusing
   `timing_closure/critical_path_model.cpp`'s `tree_reduce32` numbers),
   how much further does it push WNS negative. Compiles and runs on this
   Mac today — no Vivado needed — so these numbers are real, not TODO,
   though the per-crossing delay is a first-order approximation, not a
   datasheet quote (see caveats).
2. **Hardware pblock placement** (`slr_pblocks.tcl`): the actual Vivado
   flow — build the same checkpoint two ways, unconstrained vs. pinned to
   a single SLR via a pblock, and compare per-SLR utilization, SLR-crossing
   net counts, and WNS between the two. Real, reviewed code; unrun.

## Model caveats
`kSllDelayLowNs`/`kSllDelayHighNs` are first-order approximations, not
datasheet quotes — same framing as `critical_path_model.cpp`'s stage
delay, and for the same reason: true SLL delay depends on route
congestion and exact Laguna site usage, unknown without real
place & route. `kDspPerSlrApprox` assumes an even 3-way split of VU9P's
6840 DSP48E2 total, which is an assumption for this model, not a
measured fact — confirm against `report_utilization -slr` once on real
hardware (`slr_pblocks.tcl` produces that report).

## Results
**Crossing penalty model** (measured locally,
`clang++ -O2 -std=c++17 slr_crossing_model.cpp -o slr_crossing_model && ./slr_crossing_model`):

```
=== resource fit: does this kernel need to span SLRs at all? ===
ml_kernel_mlp    |  768 DSPs used, ~2280 DSPs/SLR budget -> 34% of one SLR (single-SLR feasible by resource count)
dot_product      |    4 DSPs used, ~2280 DSPs/SLR budget -> 0% of one SLR (single-SLR feasible by resource count)

=== crossing cost added to timing_closure's tree_reduce32 path (tree_depth=5, from critical_path_model.cpp) at a 300MHz target ===
tree_reduce32 depth=5, stage=uncongested, crossing=0.15ns | 0 crossings: delay=3.00ns WNS=+0.33ns (MEETS) | 1 crossing: delay=3.15ns WNS=+0.18ns (MEETS)
tree_reduce32 depth=5, stage=uncongested, crossing=0.30ns | 0 crossings: delay=3.00ns WNS=+0.33ns (MEETS) | 1 crossing: delay=3.30ns WNS=+0.03ns (MEETS)
tree_reduce32 depth=5, stage=congested  , crossing=0.15ns | 0 crossings: delay=4.50ns WNS=-1.17ns (FAILS) | 1 crossing: delay=4.65ns WNS=-1.32ns (FAILS)
tree_reduce32 depth=5, stage=congested  , crossing=0.30ns | 0 crossings: delay=4.50ns WNS=-1.17ns (FAILS) | 1 crossing: delay=4.80ns WNS=-1.47ns (FAILS)
```

**Hardware** — TODO: run `slr_pblocks.tcl` against a checkpoint from
`ml_kernel/ml_kernel.cpp` (or `timing_closure/pipelined_tree_reduce.cpp`)
and fill in:

| Variant | SLRs occupied | SLR-crossing nets | WNS |
|---|---|---|---|
| Unconstrained placement | TODO | TODO | TODO |
| Pinned to `pblock_slr0` | TODO | TODO | TODO |

## Design rules for multi-SLR placement
Two categories emerge from the model, and they call for opposite fixes:

1. **Avoidable crossings — pin to a single SLR.** `ml_kernel_mlp` fully
   unrolls both layers (768 multiplies total), only ~34% of one SLR's
   DSP budget by the model above — it has no resource reason to spread
   across SLRs. But the default placer optimizes for congestion/timing
   heuristically and may still spread a design across SLRs it didn't
   need to, and the crossing-cost numbers above show why that matters
   here specifically: `tree_reduce32`'s critical path (from
   `../timing_closure/`) already fails a 300MHz target by 1.17ns under
   congested routing assumptions with *zero* crossings — adding even one
   avoidable crossing pushes that another 0.15–0.30ns negative, eating
   into headroom step 15's retiming fix was trying to buy back. Anything
   this tightly coupled and small enough to fit one SLR should be
   pblock-pinned (`slr_pblocks.tcl`'s `pblock_slr0` + `CONTAIN_ROUTING`),
   not left to the placer's default judgment.
2. **Unavoidable crossings — register at the boundary, don't fight it.**
   `ddr4_bandwidth` (`../ddr4/`) touches all 4 of F1's DDR4 banks
   concurrently by design — that's the point of the step. AWS F1's shell
   places its DDR4 controllers as fixed, SLR-distributed macros outside
   user floorplan control, so a kernel with AXI masters reaching every
   bank necessarily has some of those masters crossing SLRs to get there.
   Trying to pin this kind of kernel to one SLR just relocates the
   problem (now the DDR4 controller connection itself crosses, in a place
   with even less floorplan control). The correct fix is the same
   technique `timing_closure/pipelined_tree_reduce.cpp` used for
   combinational depth, applied to a physical boundary instead of a logic
   boundary: register immediately before and after the crossing so the
   Laguna/SLL delay is absorbed by a full pipeline cycle rather than
   added onto whatever combinational logic happens to be adjacent to it
   on either side. Budget the latency; don't try to design the crossing
   away.

## Files
- `slr_crossing_model.cpp` — portable, no Vivado dependency; run it directly.
- `slr_pblocks.tcl` — the Vivado-side flow: builds unconstrained and
  pinned variants from the same checkpoint, reports per-SLR utilization
  and SLR-crossing nets for both. Discovers SLR extents from `get_slrs`
  rather than hardcoded floorplan coordinates (see file header for why).

## Hardware notes
- Required: AWS F1, Vivado 2022.x
- Run: `vivado -mode batch -source slr_pblocks.tcl -tclargs -checkpoint <post_synth.dcp> -pin_pattern "*ml_kernel_mlp*" -outdir <dir>`
