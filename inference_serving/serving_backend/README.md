# serving_backend

**Status: code-complete AND locally run — pure CPU router logic, no GPU
dependency. Real CPU backend, real (honestly-unavailable) GPU/FPGA/TPU/NPU
backend registrations.**

## What this measures

PLAN.md Phase 9 step 8: inference serving layer works with CPU, GPU,
FPGA, and TPU backends via a unified router. Extended 2026-07-30 (Phase 15
step 7 / SCOPE.md's "NPU as a fifth ServingRouter backend" note) to
register NPU as a fifth backend, same convention.

## Design

- `ServingRouter`: a registry of `Backend -> (BackendInfo, GenerateFn)`.
  `route(preferred, ...)` dispatches directly if `preferred` is
  registered, available, and has a generate function; otherwise falls
  back through a fixed priority order (GPU, TPU, FPGA, NPU, CPU) to the
  first backend that qualifies, and throws if none do.
- Mirrors `compiler/dialect`'s `DeviceKind` enum in spirit
  (CPU/GPU/FPGA/TPU/NPU/Unassigned) but doesn't depend on it — that enum
  lives behind Phase 4's `MLIR_DIR` gate; this router has no MLIR
  dependency at all, so it builds and runs everywhere Phase 4 doesn't.
- **Only CPU can actually execute on this Mac** — `make_cpu_backend()` is
  a real backend, greedy-decoding via `transformer::model_forward`,
  identical to `transformer_test.cpp`'s own generation loop (checked
  directly in `test_cpu_backend_matches_direct_greedy_generation`).
  GPU/FPGA/TPU/NPU are registered with `available = false` and an honest
  reason string (`"CUDA not found..."`, `"XILINX_VITIS not set"`, `"no
  TPU device"`, `"no NPU/ANE toolchain (coremltools/onnxruntime not
  installed)"`) — the same "real entry, hardware-gated" convention this
  repo's build system uses everywhere else, rather than just omitting
  them from the table. That matters for testing: with real (if
  unavailable) GPU/FPGA/TPU/NPU entries, the fallback *priority order*
  itself is testable — `test_priority_order_prefers_higher_priority_fallback`
  registers a stub TPU backend alongside CPU and confirms a request for
  the (unavailable) FPGA backend falls back to TPU, not CPU, per the
  fixed priority list, not just "falls back to whatever's available."
- **NPU is placed last among the four accelerators** (after FPGA, before
  CPU) rather than competing with GPU/TPU/FPGA for top priority — it's
  inference-only, edge/mobile-first, restricted-operator-model hardware
  (see `npu_engine/op_coverage`'s real finding), a categorically different
  kind of accelerator than the three general datacenter backends already
  registered. `test_npu_outranks_cpu_but_not_fpga_in_fallback` proves both
  halves of that ranking claim directly (beats CPU, loses to FPGA), not
  just asserted in a comment.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac;
re-run 2026-07-30 after adding NPU, zero regressions)

```
PASS  falls back to CPU when GPU is unavailable
PASS  reports that a fallback occurred
PASS  generated prompt + max_new_tokens tokens
PASS  uses the preferred backend directly
PASS  reports no fallback when the preferred backend qualifies
PASS  prefers TPU over CPU as the fallback, per priority order
PASS  actually dispatched to TPU's generate function, not CPU's
PASS  prefers NPU over CPU as the fallback, per priority order
PASS  actually dispatched to NPU's generate function, not CPU's
PASS  prefers FPGA over NPU as the fallback, per priority order
PASS  throws when no registered backend is both available and has a generate function
PASS  router's CPU backend produces identical output to direct greedy generation
PASS
```

## serving_daemon — the real long-running process (added 2026-07-30)

`k8s/serving/deployment.yaml` (Phase 16 step 4) was written to point at
"the serving process" before one actually existed — `ServingRouter`
previously only lived inside `serving_router_test.cpp`'s `main()`, which
runs a fixed set of assertions and exits. `serving_daemon.cpp` closes that
gap: a real, long-running process.

- Trains the same tiny transformer `transformer_test.cpp`'s
  `test_trains_and_generates()` validates (identical corpus/config/
  training loop — a known-good recipe, not a new untested one), then
  builds a `ServingRouter` the same way `serving_router_test.cpp` does
  (CPU real and available, GPU/FPGA/TPU registered `available=false`).
- Opens a real TCP listening socket (`SERVING_PORT` env var, default
  8080) and serves a deliberately minimal line protocol: one line of
  plain text in (a prompt), one line of greedy-generated text back —
  proving the process/socket/routing model works, not building a
  production wire format.
- Guards against a real bug found while testing this: `model_forward`
  indexes positional embeddings by absolute sequence position with no
  bounds check beyond an `assert()` (a process abort, not a catchable
  exception) once position >= `max_seq_len`. A prompt long enough that
  `prompt.size() + max_new_tokens > max_seq_len` used to take the whole
  daemon down on the first oversized request. Fixed by clamping
  `max_new_tokens` per-request to what still fits, and rejecting
  (cleanly, per-connection, not process-fatal) any prompt that's already
  at or past `max_seq_len` on its own. `max_seq_len` was also bumped
  32 -> 64 for headroom beyond the 21-character training corpus.
- Handles `SIGTERM`/`SIGINT` via a `select()`-with-timeout accept loop
  (checked every 1s) instead of blocking in `accept()` forever — a K8s
  pod gets `SIGTERM` before `SIGKILL` on shutdown, and this exits cleanly
  within that grace period rather than being forcibly killed mid-request.

### Real transcript (captured 2026-07-30, this Mac, `SERVING_PORT=18081 SERVING_MAX_NEW_TOKENS=20`)

```
$ printf "the quick fox " | nc -w 5 127.0.0.1 18081
the quick fox jumps qumps qu qux j

$ printf "the " | nc -w 5 127.0.0.1 18081
the quick fox jumps qump

$ printf "xyz123" | nc -w 5 127.0.0.1 18081
ERROR: CharTokenizer::encode: character not in vocabulary

$ python3 -c "print('the quick fox jumps ' * 5, end='')" | nc -w 5 127.0.0.1 18081
ERROR: prompt too long for this model's max_seq_len

# daemon still alive after both error requests (pgrep confirms one process)
$ kill -TERM <pid>
# server log:
serving_daemon: training demo model on corpus "the quick fox jumps "...
serving_daemon: model trained, router ready (CPU available; GPU/FPGA/TPU registered unavailable)
serving_daemon: listening on 0.0.0.0:18081 (max_new_tokens=20)
serving_daemon: shutting down cleanly
```

The generated continuation past "jumps " degrades (`qumps qu qux j`) since
that's beyond what a 400-epoch fit on a 21-character corpus actually
memorized — expected, and not the point of this test: the point is a real
process, real socket, real router dispatch, and a real trained model
round-tripping correctly, including the two error paths staying
non-fatal to the process.

## Hardware notes
- GPU: only added by the build when a CUDA toolchain is present (not the
  case here — see `flash_decoding/README.md`'s identical gate).
- FPGA: needs Vivado/Vitis HLS toolchain (`fpga_engine/`'s gate).
- TPU: needs a GCP TPU VM + JAX (`tpu_engine/`'s gate).
- NPU: needs real ANE/Coral hardware + `coremltools`/`onnxruntime`
  (`npu_engine/`'s gate — see `npu_engine/README.md`).
- A real deployment would register each backend's actual
  `GenerateFn` (calling into `gpu_engine`/`fpga_engine`/`tpu_engine`/
  `npu_engine`'s kernels) once built on that hardware — the router itself
  needs no changes.
