# timing_closure — critical path analysis and retiming

**Status: hardware timing closure is code complete and unrun (F1 required);
the combinational-depth model that motivates the fix is measured locally.**

## What this measures
Two things, deliberately kept separate since only one needs an F1 instance:

1. **Analytical model** (`critical_path_model.cpp`): a first-order,
   locally-computed comparison of a flat O(N) accumulate chain vs. a
   balanced O(log2 N) binary tree, for `ml_kernel_mlp`'s two reduction
   widths (layer1: N=16, layer2: N=32 — see `fpga_engine/ml_kernel/`).
   `ml_kernel_mlp` writes its reduction as a single unrolled expression
   (`acc += in[i] * w[i][h]`), which Vitis HLS may or may not schedule as
   a balanced tree on its own. This model predicts which of the two
   structures is the likely worst path and quantifies the Fmax gain
   retiming it should buy, without needing Vivado to find out. It
   compiles and runs on this Mac today — no HLS/Vivado toolchain needed
   — so its numbers are real, not TODO, even though they're an
   engineering approximation (see caveats below), not a silicon
   measurement.
2. **Hardware timing closure** (`close_timing.tcl`, `pipelined_tree_reduce.cpp`):
   the actual post-route flow — top-10 critical path report, Vivado's own
   QoR-suggestion analysis, retiming (`phys_opt_design -retime`), and a
   placement-effort fallback for route-dominated paths — plus the
   source-level retiming fix (explicit staged tree reduction) the model
   motivates. Both are real, reviewed code; neither has run against a
   real checkpoint.

## Model caveats
`kMultiplyDelayNs`, `kStageDelayLowNs`, `kStageDelayHighNs` are first-order
engineering approximations, not datasheet quotes — a DSP48E2
multiply-accumulate and a LUT-fabric adder stage's actual delay depend on
speed grade, operand width, and post-place routing congestion, none of
which are known without synthesizing on real hardware. What the model
does *not* depend on getting exactly right is the relative comparison:
whatever the true per-stage delay is, an N-term flat chain is always
`(N-1)` stages deep and a balanced tree is always `ceil(log2 N)` stages
deep — for N=32 that's 31 vs. 5. Two delay points (`uncongested` /
`congested`) are reported specifically so the finding doesn't hinge on
picking one convenient number.

## Results
**Analytical model** (measured locally,
`clang++ -O2 -std=c++17 critical_path_model.cpp -o critical_path_model && ./critical_path_model`):

```
=== flat accumulate chain vs. balanced binary tree ===
layer1  (N=16) | stage=0.30ns (uncongested) | flat: depth=15 delay= 6.00ns Fmax= 167MHz | tree: depth= 4 delay= 2.70ns Fmax= 370MHz | speedup=2.22x
layer1  (N=16) | stage=0.60ns (congested  ) | flat: depth=15 delay=10.50ns Fmax=  95MHz | tree: depth= 4 delay= 3.90ns Fmax= 256MHz | speedup=2.69x
layer2  (N=32) | stage=0.30ns (uncongested) | flat: depth=31 delay=10.80ns Fmax=  93MHz | tree: depth= 5 delay= 3.00ns Fmax= 333MHz | speedup=3.60x
layer2  (N=32) | stage=0.60ns (congested  ) | flat: depth=31 delay=20.10ns Fmax=  50MHz | tree: depth= 5 delay= 4.50ns Fmax= 222MHz | speedup=4.47x

=== timing closure at a 300MHz target clock ===
layer1  @ 300MHz (period=3.33ns, stage=uncongested) | flat WNS= -2.67ns (FAILS) | tree WNS= +0.63ns (MEETS)
layer1  @ 300MHz (period=3.33ns, stage=congested  ) | flat WNS= -7.17ns (FAILS) | tree WNS= -0.57ns (FAILS)
layer2  @ 300MHz (period=3.33ns, stage=uncongested) | flat WNS= -7.47ns (FAILS) | tree WNS= +0.33ns (MEETS)
layer2  @ 300MHz (period=3.33ns, stage=congested  ) | flat WNS=-16.77ns (FAILS) | tree WNS=-1.17ns (FAILS)
```

The finding is more nuanced than "tree always wins": at the optimistic
(`uncongested`) delay assumption, a combinational tree alone closes a
300MHz target for both layers. At the pessimistic (`congested`)
assumption, the tree *still fails* — narrowing the gap from 7–17ns to
0.6–1.2ns, but not closing it. That's why `pipelined_tree_reduce.cpp`
doesn't stop at "fewer logic levels": it restructures the reduction into
explicit per-level arrays, giving Vitis HLS's scheduler concrete
statement boundaries to register against under real (post-place)
congestion, not just a shallower combinational expression that might
still need registering mid-tree.

**Hardware** — TODO: synthesize `ml_kernel_mlp` (flat) and
`pipelined_tree_reduce.cpp` (staged tree) on F1, run `close_timing.tcl`
against any checkpoint that fails synth.tcl's WNS gate, and fill in:

| Kernel | WNS before | WNS after -retime | WNS after placement fallback | Fmax achieved |
|---|---|---|---|---|
| `ml_kernel_mlp` (flat) | TODO | TODO | TODO | TODO |
| `tree_reduce16`/`tree_reduce32` (staged) | TODO | TODO | TODO | TODO |

Top-10 critical path classification (from `report_timing`) and which fix
addressed each: TODO — fill in from `reports/top10_critical_paths.rpt`
once `close_timing.tcl` has run.

## Files
- `critical_path_model.cpp` — portable, no HLS dependency; run it directly.
- `pipelined_tree_reduce.cpp` — the source-level retiming fix (HLS,
  unrun). Deliberately standalone, not folded into `ml_kernel.cpp` — see
  the file header for why.
- `close_timing.tcl` — the Vivado-side flow: top-10 report, QoR
  suggestions, retiming, placement fallback, WNS gate. Mirrors
  `tcl_pipeline/synth.tcl`'s argument/exit-code conventions.

## Hardware notes
- Required: AWS F1, Vitis HLS + Vivado 2022.x
- Run against any checkpoint `tcl_pipeline/synth.tcl` produced that
  failed its WNS >= 0 gate:
  `vivado -mode batch -source close_timing.tcl -tclargs -checkpoint <post_route.dcp> -outdir <dir>`
