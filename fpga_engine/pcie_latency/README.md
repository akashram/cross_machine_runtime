# pcie_latency — PCIe latency component decomposition

**Status: code complete — requires XRT + a real xclbin on AWS F1 to run.**

## What this measures
A single kernel invocation's latency bundles together at least four
physically distinct costs. `pcie_latency.cpp` measures each in isolation
rather than only the bundled total (which `dma/dma_orchestration.cpp`
measures):

1. **BAR write (doorbell)**: `xrt::ip::write_register()` — a single MMIO
   register write, no DMA, no interrupt. Isolates pure PCIe write latency.
2. **DMA descriptor processing + data transfer**: `bo.sync()` timed at 5
   buffer sizes (4KB to 64MB), then fit with ordinary least squares to
   `time = descriptor_overhead + size / bandwidth`. The regression is the
   point — it separates the fixed per-transfer cost (intercept) from the
   size-dependent cost (slope), rather than assuming which one dominates
   for a given transfer size.
3. **Interrupt dispatch overhead**: `run.wait()` timed on the
   cheapest possible kernel invocation, so the measurement is dispatch
   latency alone, not dispatch-plus-real-work.
4. **Poll dispatch overhead**: same isolation, but for the `run.state()`
   polling loop at a fixed interval — comparable directly against (3).

## Results
TODO: run on F1 against a real xclbin.

| Component | Latency |
|---|----|
| BAR write (doorbell) | TODO ns |
| DMA descriptor overhead (fixed) | TODO ns |
| DMA bandwidth (slope) | TODO GB/s |
| Interrupt dispatch overhead | TODO ns |
| Poll dispatch overhead (10us interval) | TODO ns |

## Hardware notes
- Required: AWS F1, XRT installed, a built `.xclbin`
- Run: `./pcie_latency <xclbin> <kernel_name>`
