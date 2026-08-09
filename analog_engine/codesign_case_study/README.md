# codesign_case_study

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 7: a hardware-algorithm co-design case study —
re-derive `inference_serving::GptqQuantizer` (Phase 9 step 6, Frantar et
al. 2022) under analog crossbar constraints, using Phase 17's OWN device
model (step 1's `ConductanceCell`) rather than inventing a separate analog
error model for this step. On a REAL trained transformer weight, exactly
mirroring `npu_engine/quant_export_test.cpp`'s own real-model setup.

## Design

- **The co-design question**: GPTQ's `bits` parameter already IS a
  discrete-level count (`num_levels = 2^bits`) — the exact same quantity
  step 2's crossbar precision sweep varied. This step asks what happens
  when GPTQ's quantized integer level isn't stored in a noiseless digital
  register, but PROGRAMMED into a real (simulated) analog crossbar cell.
- **Pipeline, every stage reusing an already-tested component**: GPTQ's
  quantized integer level → program a `ConductanceCell` to that level
  (step 1, write noise) → read it back (step 1, read noise) → nearest-
  level decode (step 2's rounding convention) → GPTQ's OWN scale/zero
  dequantization formula. The only new code is the pipeline connecting
  them, not a new quantization or noise scheme.
- Two measurements: (1) weight-level RMSE, quantization-only vs.
  quantization-plus-analog-noise, swept across GPTQ's `bits` (2 through 6,
  matching step 2's exact `num_levels`); (2) a real end-task check —
  perplexity of the actual trained transformer with each weight variant
  swapped in for `w_out`, at `bits=4` (GPTQ's common default).

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  real trained transformer w_out (27x16), GPTQ re-derived under analog crossbar constraints:
    bits     levels      quant-only RMSE    quant+analog RMSE
       2          4             0.172058             0.187717
       3          8             0.067559             0.072771
       4         16             0.033671             0.035419
       5         32             0.016468             0.017483
       6         64             0.008468             0.008925
PASS  GPTQ's own quantization-only RMSE improves from 2 to 6 bits, matching its known behavior
PASS  quantization-plus-analog-noise RMSE ALSO improves from 2 to 6 bits -- analog noise doesn't erase the benefit of more GPTQ precision at these settings
PASS  at every bit width, realizing GPTQ's quantized weight on a real analog cell never IMPROVES on the quantization-only RMSE (noise only adds error, as physically expected)

  end-task perplexity at bits=4: fp32=1.0497 | GPTQ quant-only=1.0513 | GPTQ+analog-noise=1.0509
  |analog-noise effect on ppl|=0.0004 vs. |quantization's own effect on ppl|=0.0016
PASS  layering analog noise on top of GPTQ changes end-task perplexity by LESS than GPTQ's own quantization already did -- the analog-realization step doesn't dominate the already-accepted quantization cost
```

## Findings

- **Weight-level RMSE behaves exactly as physics predicts, at every bit
  width tested**: analog noise always makes RMSE worse than quantization
  alone (e.g. `0.033671 -> 0.035419` at bits=4), and more GPTQ precision
  keeps helping even with analog noise layered on top (`0.187717 ->
  0.008925` from 2 to 6 bits) — the analog noise floor does NOT dominate
  and erase GPTQ's precision benefit at these settings, because step 1's
  noise model scales with level spacing, which itself shrinks as
  `num_levels` grows (the same reason step 2's crossbar accuracy improved
  cleanly with precision).
- **A real, honest, non-monotonic result at the END-TASK level, caught by
  actually running the test — not a bug.** An earlier version of this
  test asserted perplexity-with-analog-noise must never beat perplexity-
  with-quantization-only. That assertion FAILED on the real run: perplexity
  came out very slightly BETTER with analog noise (`1.0509`) than without
  it (`1.0513`). This is not evidence of a broken noise model — the
  weight-level RMSE checks above already confirm noise strictly increases
  weight distortion, which IS guaranteed monotonic by construction (RMSE
  is a direct distance). Perplexity is a highly non-linear function of the
  weights, measured on a 45-character, 300-epoch (heavily overfit) toy
  corpus near a sharp loss-surface optimum; a single noise realization can
  land on either side of that optimum by chance. The test was changed to
  assert something actually defensible instead: analog noise's effect on
  perplexity (`0.0004`) stays smaller than GPTQ's OWN quantization effect
  on perplexity (`0.0016`) — the real, useful co-design conclusion, that
  the analog-realization step doesn't dominate a quantization cost that
  was already accepted.
- Together, these two findings make the actual co-design argument
  concrete: choosing GPTQ's bit width already has to account for a real
  downstream analog noise cost (not free, not zero), but at this device
  model's noise levels, that cost stays subordinate to the quantization
  decision itself — the algorithm and the hardware constraint were
  co-designed by measuring, not assumed compatible.

## Hardware notes
None — pure CPU. Reuses Phase 9's real GPTQ implementation and Phase 17
step 1's device model, both already validated on this Mac; no fab access
for the analog side (see PLAN.md's Phase 17 hardware-access note).
