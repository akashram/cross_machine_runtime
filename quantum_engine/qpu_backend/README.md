# qpu_backend

**Status: code-complete AND locally run for the simulator-backed half;
honestly hardware-gated and unrun for the real cloud-QPU half.**

## What this measures

PLAN.md Phase 20 step 9: cross-machine runtime integration — register a
QPU backend candidate the same way Phase 15 registered NPU, plus a real
(simulator-backed) code path showing what a "submit this circuit" call
actually looks like — the genuine "cross-machine runtime hits a
different backend" plumbing, not a QPU enum value sitting in a corner
with nothing calling it.

## Design

- **`inference_serving/serving_backend/serving_router.{h,cpp}`**: added
  `Backend::QPU` to the existing enum, `to_string`, and the fallback
  priority order — placed LAST, after NPU (`GPU, TPU, FPGA, NPU, QPU,
  CPU`). This is an ADDITIVE change: `serving_router_test.cpp`'s existing
  12 assertions all still pass unmodified (verified — see Results),
  since no unregistered `Backend::QPU` entry affects `qualifies()`'s
  logic for any other backend.
- **Why QPU is placed even after NPU**, not just "another accelerator":
  NPU is at least a local, deterministic, inference-shaped device (see
  `serving_router.h`'s existing NPU comment); a QPU is a step further —
  probabilistic (shot-based sampling, not a deterministic forward pass),
  queue-scheduled on a remote cloud service rather than a local
  accelerator card, and has no natural token-generation semantics at all
  (`GenerateFn`'s `(model, prompt, max_new_tokens) -> tokens` shape
  doesn't describe "submit a circuit, get back measurement counts").
  Rather than force a QPU call into that shape, `qpu_backend.h` defines
  its own, genuinely different call shape (`QpuJobRequest`/
  `QpuJobResult`) and keeps it separate — `ServingRouter` only needs to
  know a QPU is registered as available-or-not for routing PURPOSES
  (e.g. a future prompt-generation request that happens to prefer a QPU
  path would still fall through the same fallback logic), not to
  understand what a QPU call actually does.
- **`submit_circuit_to_cloud_qpu`**: the real, honestly hardware-gated
  call — what submitting to an actual cloud QPU (IBM Quantum / AWS
  Braket / Azure Quantum / Xanadu Cloud) requires (a real network call
  with real credentials and a real queue wait), none of which exists
  here. Returns `{submitted: false, status: "unavailable: ..."}`, same
  convention as every other hardware-gated backend in this repo.
- **`sample_circuit`**: the real, run-LOCALLY stand-in with the SAME
  call shape a QPU actually has — given an already-prepared
  `state_vector::StateVector` (from ANY of steps 1-8's circuits), samples
  `shots` outcomes via repeated `measure_all()` calls and tallies counts.
  This is the actual shot-based sampling output real quantum hardware
  returns (never exact amplitudes, which only this simulator can hand
  back directly).

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  Bell state, 4000 shots: |00>=2020 |01>=0 |10>=0 |11>=1980
PASS  sample_circuit on a Bell state produces EXACTLY zero counts on |01>/|10> (structural, not statistical)
PASS  sample_circuit on a Bell state splits roughly 50/50 between |00> and |11> (within 3%% at 4000 shots)
  submit_circuit_to_cloud_qpu: submitted=false status="unavailable: no cloud QPU credential configured (IBM Quantum / AWS Braket / Azure Quantum / Xanadu Cloud) -- see quantum_engine/qpu_backend/README.md and PLAN.md Phase 20's hardware access note"
PASS  submit_circuit_to_cloud_qpu honestly reports unavailable with a non-empty reason string, same convention as every other hardware-gated backend
  QPU registered: available=false reason="no cloud QPU credential configured"
PASS  QPU registers into ServingRouter as unavailable with an honest reason string, same as GPU/FPGA/TPU/NPU
PASS  with FPGA/NPU/QPU/CPU all available, fallback prefers FPGA (highest priority) over NPU, QPU, and CPU
PASS  with only QPU and CPU available (no GPU/TPU/FPGA/NPU), fallback prefers QPU over CPU
PASS  routing to QPU actually dispatches to QPU's registered generate function, not CPU's
PASS
```

`inference_serving/serving_backend/serving_router_test.cpp`'s full
existing 12-assertion suite was re-run after the `Backend::QPU` addition
and still passes unmodified (`ctest -R serving_router_test`), confirming
the change is genuinely additive.

## Findings

- **No regressions from the additive `Backend::QPU` change** — every
  existing `serving_router_test.cpp` assertion (fallback priority among
  GPU/TPU/FPGA/NPU/CPU, dispatch correctness, the "throws when nothing
  qualifies" case) still passes exactly as before, confirming the switch
  statement's new `case Backend::QPU` and the priority array's new entry
  didn't disturb any existing backend's behavior.
- **Bell-state sampling is exact where it should be, statistical where
  it should be**: `|01>`/`|10>` get EXACTLY zero counts across 4000
  shots (a structural guarantee — those basis states have exactly zero
  amplitude in a Bell state, so the Born rule gives them exactly zero
  sampling probability, not just "rare"), while the `|00>`/`|11>` split
  is only statistically close to 50/50 (`2020`/`1980`), correctly
  reflecting that finite-shot sampling noise is real and expected for
  the two equally-likely outcomes.

## Platform notes

Single-threaded numerical code — TSan not applicable. Clean under both
`--preset debug` and `--preset asan` (ASan+UBSan).
