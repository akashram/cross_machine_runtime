# Analog & Unconventional Compute Hardware — Design

## 1. Why this phase has no "portable model + hardware-gated kernel" split at all

Phases 3/7/8/15 all write real code for every step and defer running the
hardware-touching half to a cloud/edge instance that doesn't exist on this
Mac. Phase 17 can't even offer that deferral: there is no cloud provider,
spot instance, or ~$60 USB accelerator that rents analog/neuromorphic
silicon the way GPU/FPGA/TPU/NPU can (eventually) be rented or bought.
Every constant this phase needs that would normally come from characterizing
real hardware — device noise magnitudes, NVM endurance/retention/energy
figures, ADC energy per bit, DRAM access energy, bitline parasitic R/C — has
to come from the literature instead, explicitly labeled as illustrative
order-of-magnitude figures, not measurements. That's not a shortcut; it's
the honest ceiling on what any of these eight steps can claim without fab
access, stated in each step's own README rather than hidden in a
methods section.

## 2. The reuse chain: eight steps building on each other, not eight separate models

The step order matters and each step deliberately reuses the previous
ones' real components:

- Step 1 (`device_model`) is the only step that invents new noise/drift/
  endurance numbers. Every later step that needs analog imprecision reuses
  THIS model, not a new one.
- Step 2 (`crossbar_mac`) is the first consumer: it programs step 1's
  `ConductanceCell`s with GPTQ-style differential encoding and reads them
  back with step 1's noise, rather than adding its own error term.
- Step 4 (`energy_model`) reuses `npu_engine/cost_model`'s existing CPU/
  GPU/NPU TOPS/W constants directly — the analog-vs-digital comparison is
  against numbers this repo already committed to in Phase 15, not a fresh
  set of digital-device guesses that could quietly diverge from them.
- Step 5 (`dataflow_model`) is reused unmodified by step 6 (sweeping PE
  array size on real GEMM shapes) — the same `analyze_dataflow` function,
  no copy-and-modify.
- Step 6 uses REAL shapes from `transformer/transformer_test.cpp`'s
  actual trained config (`d_model=16, num_heads=2, d_ff=32, seq_len=32,
  vocab_size=16`) — not invented GEMM dimensions, so its utilization
  numbers are about a model this repo actually trained, not a synthetic
  stand-in.
- Step 7 (`codesign_case_study`) is the phase's clearest demonstration of
  the reuse discipline: it pipelines Phase 9's real `GptqQuantizer`
  (unmodified) through step 1's real `ConductanceCell` (unmodified) on a
  real trained transformer weight — the only new code is the ~80-line
  pipeline connecting three already-tested components, not a fourth
  quantization or noise scheme.
- Step 8 (`circuit_transient`) explicitly closes a gap step 2 left open
  (instantaneous reads) rather than being an unrelated add-on.

## 3. Two real bugs, both caught by running the tests, both worth keeping visible

**Step 1's endurance model** originally rolled a fresh Bernoulli
stuck-check on every single write, using the cumulative failure-
probability curve evaluated at the current cycle count. That's wrong: it
turns one population-level failure probability into dozens of
near-independent chances to fail before reaching a given cycle count. At
100 writes (100x below a 1000-cycle rated endurance), this produced a 50%
stuck rate — nonsensical for a cell nowhere near its rated endurance. The
fix: draw ONE failure-threshold percentile per cell at construction, and
compare the cumulative curve against that fixed percentile on each write
instead of re-rolling. Same test then reported 4.0% at 100 writes and
100% at 100,000 writes — a coherent reliability curve instead of a
compounding hazard.

**Step 7's original perplexity assertion** claimed analog noise could
never IMPROVE end-task perplexity relative to quantization-only. That
assertion failed on the real run: perplexity came out very slightly
better with analog noise (1.0509 vs. 1.0513). This was NOT a bug in the
noise model — step 7's own weight-level RMSE checks, which ARE guaranteed
monotonic by construction (RMSE is a direct distance; noise cannot reduce
it), confirmed noise always increases weight distortion at every bit
width tested. The perplexity result is a real, non-monotonic artifact of
measuring a highly non-linear task metric on a heavily overfit
(300-epoch, 45-character corpus) toy model near a sharp loss-surface
optimum — a single noise realization can land on either side of that
optimum by chance. The assertion was replaced with one that IS
defensible: analog noise's perplexity effect stays smaller than GPTQ's
own quantization effect, which is the actually useful co-design
conclusion (the analog-realization step doesn't dominate an already-
accepted quantization cost).

Both bugs are left documented in their step's own README and (for step 1)
in the source comment, not quietly fixed and forgotten — the same
discipline this repo's earlier phases apply throughout (e.g. the
ring-allreduce chunk-ownership off-by-one in Phase 5, the DMA controller
timing bug in Phase 7 step 20).

## 4. Two independently-built models converging on the same conclusion

Step 6's systolic-array sweep and `tpu_engine/mxu_opt` (Phase 8) reach the
same qualitative finding — PE utilization is a function of the RATIO
between workload and array dimensions, not either one alone — from
opposite directions. `mxu_opt` fixes real TPU hardware (its actual
128x128 MXU) and sweeps workload alignment; step 6 fixes several real
workload shapes and sweeps hardware array size. Neither was built with
the other's numbers in mind. The same kind of convergence happens between
step 2 (crossbar size doesn't help MAC accuracy — a statistics argument)
and step 8 (crossbar size actively hurts settling time/bandwidth — a
circuit-physics argument): two completely different mechanisms, same
"bigger isn't free" conclusion. Neither convergence was engineered; both
are worth noting because they weren't.

## 5. Disclosed simplifications, by step

- **Step 2**: signed weights via differential G+/G- cell pairs (the real
  ISAAC/PRIME technique, not a shortcut).
- **Step 3**: every NVM comparison number is a literature-informed
  representative point within a cited range, not a measurement; the
  composite figure-of-merit is an explicitly illustrative geometric mean,
  not a validated accelerator-design cost model.
- **Step 4**: the ADC-per-bit energy constant (`0.3 pJ/bit`) is
  illustrative; the specific NPU-crossover finding is sensitive to that
  constant even though the qualitative shape (ADC cost scales with
  precision) isn't.
- **Step 5**: Row-Stationary is a GEMM-adapted simplification of
  Eyeriss's actual convolution-specific row mapping, disclosed explicitly
  rather than presented as a literal reproduction; K-tiling was added
  specifically so Output-Stationary's real advantage (avoiding partial-
  sum round trips) shows up numerically at all.
- **Step 8**: the RC line model is lumped (`tau = R_total * C_total`)
  rather than a full Elmore-delay-factor treatment (`0.5*R*C` for a truly
  distributed line); the quadratic size-scaling conclusion holds either
  way, only the absolute constant would shift.

## 6. What this phase does and doesn't claim

This phase claims: a coherent, internally-consistent, real (compiled and
run) set of models that reproduce several qualitatively-correct,
literature-consistent behaviors of analog compute-in-memory hardware —
noise/drift/endurance tradeoffs, precision-vs-size effects, dataflow
reuse patterns, ADC energy dominance, and RC settling-time scaling — and
connects them to this repo's own already-real algorithms (GPTQ) and
models (`transformer/`, `npu_engine/cost_model`, `tpu_engine/mxu_opt`).

It does NOT claim: that any specific number here (pJ/MAC, endurance
cycles, ADC energy, settling time) is what a real fabricated RRAM
crossbar chip would measure. No fab access exists to check that, and
every step says so in its own README.
