# dma — host <-> FPGA DMA orchestration via XRT

**Status: code complete — requires XRT + a real xclbin on AWS F1 to run.**

## What this measures
`dma_orchestration.cpp` is real host-side code against XRT's native C++
API (`xrt::device`, `xrt::bo`, `xrt::kernel`), not a stub — buffer
allocation, host-to-device sync (`XCL_BO_SYNC_BO_TO_DEVICE`), kernel
launch, device-to-host sync, and two completion-wait strategies against
the same dot_product-shaped kernel:

- `run_interrupt_variant`: `xrt::run::wait()` blocks on the kernel-done
  interrupt. No host CPU spent spinning; pays interrupt dispatch latency
  (context switch + ISR) before the host observes completion.
- `run_poll_variant`: repeatedly checks `run.state()` at a configurable
  poll interval instead of blocking. Burns host CPU on the polling
  thread; expected to win on latency for short kernels (where interrupt
  overhead is a larger fraction of total time) and lose on CPU
  utilization for long-running ones.

Both variants share identical setup/DMA code (`setup()`, `fill_inputs()`)
so the only difference measured between them is the completion-wait
mechanism — same pattern as loop_opt/dsp_lut isolating one variable at a
time.

## Results
TODO: run on F1 against a real xclbin (dot_product's, once synthesized).

| Variant | End-to-end latency (host call -> result) | Host CPU during wait |
|---|----|----|
| Interrupt (`run.wait()`) | TODO | TODO |
| Poll, 10us interval | TODO | TODO |
| Poll, 100us interval | TODO | TODO |

## Hardware notes
- Required: AWS F1, XRT installed, a built `.xclbin` (e.g. dot_product's)
- Link against `libxrt_coreutil.so` (part of the XRT install)
