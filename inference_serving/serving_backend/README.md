# serving_backend

**Status: code-complete AND locally run — pure CPU router logic, no GPU
dependency. Real CPU backend, real (honestly-unavailable) GPU/FPGA/TPU
backend registrations.**

## What this measures

PLAN.md Phase 9 step 8: inference serving layer works with CPU, GPU,
FPGA, and TPU backends via a unified router.

## Design

- `ServingRouter`: a registry of `Backend -> (BackendInfo, GenerateFn)`.
  `route(preferred, ...)` dispatches directly if `preferred` is
  registered, available, and has a generate function; otherwise falls
  back through a fixed priority order (GPU, TPU, FPGA, CPU — fastest-to-
  slowest in a real deployment) to the first backend that qualifies, and
  throws if none do.
- Mirrors `compiler/dialect`'s `DeviceKind` enum in spirit
  (CPU/GPU/FPGA/TPU/Unassigned) but doesn't depend on it — that enum
  lives behind Phase 4's `MLIR_DIR` gate; this router has no MLIR
  dependency at all, so it builds and runs everywhere Phase 4 doesn't.
- **Only CPU can actually execute on this Mac** — `make_cpu_backend()` is
  a real backend, greedy-decoding via `transformer::model_forward`,
  identical to `transformer_test.cpp`'s own generation loop (checked
  directly in `test_cpu_backend_matches_direct_greedy_generation`).
  GPU/FPGA/TPU are registered with `available = false` and an honest
  reason string (`"CUDA not found..."`, `"XILINX_VITIS not set"`, `"no
  TPU device"`) — the same "real entry, hardware-gated" convention this
  repo's build system uses everywhere else, rather than just omitting
  them from the table. That matters for testing: with real (if
  unavailable) GPU/FPGA/TPU entries, the fallback *priority order* itself
  is testable — `test_priority_order_prefers_higher_priority_fallback`
  registers a stub TPU backend alongside CPU and confirms a request for
  the (unavailable) FPGA backend falls back to TPU, not CPU, per the
  fixed priority list, not just "falls back to whatever's available."

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  falls back to CPU when GPU is unavailable
PASS  reports that a fallback occurred
PASS  generated prompt + max_new_tokens tokens
PASS  uses the preferred backend directly
PASS  reports no fallback when the preferred backend qualifies
PASS  prefers TPU over CPU as the fallback, per priority order
PASS  actually dispatched to TPU's generate function, not CPU's
PASS  throws when no registered backend is both available and has a generate function
PASS  router's CPU backend produces identical output to direct greedy generation
PASS
```

## Hardware notes
- GPU: only added by the build when a CUDA toolchain is present (not the
  case here — see `flash_decoding/README.md`'s identical gate).
- FPGA: needs Vivado/Vitis HLS toolchain (`fpga_engine/`'s gate).
- TPU: needs a GCP TPU VM + JAX (`tpu_engine/`'s gate).
- A real deployment would register each backend's actual
  `GenerateFn` (calling into `gpu_engine`/`fpga_engine`/`tpu_engine`'s
  kernels) once built on that hardware — the router itself needs no
  changes.
