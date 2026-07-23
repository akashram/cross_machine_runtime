# tcl_pipeline

**Status: code complete — requires AWS F1 (Vivado 2022.x) to run.**

## What this builds
A fully scriptable synthesis → implementation → bitstream flow with no GUI
dependency, so it can run headless in CI. `synth.tcl` is generic and
parameterized (`-top`, `-ip_dir`, `-xdc`, `-outdir`) — every later kernel
step (dot_product, loop_opt, ml_kernel, ...) calls it through
`run_pipeline.sh` instead of duplicating the Vivado flow per kernel.

Pipeline stages, all in one `vivado -mode batch` invocation:
1. Project creation, targeting the F1 VU9P part (`xcvu9p-flgb2104-2-i`)
2. IP import (HLS-exported `.xci`/IP-XACT `.zip`) + constraints
3. `synth_design`, with post-synthesis utilization report
4. `opt_design` → `place_design` → `route_design`, with post-place utilization
5. `write_bitstream`
6. Timing, utilization, power, and DRC reports (always generated, even on
   a failing build, so CI can archive them for debugging)
7. **Timing closure gate**: exits non-zero if worst negative setup slack is
   below 0 ns — a bitstream that doesn't close timing is treated as a
   build failure, not a warning, so it can't silently pass CI.

## Results
TODO: run on F1 hardware.

| Metric | Value |
|--------|-------|
| Build time (synth+impl+bitstream) | TODO |
| WNS on first real kernel build | TODO |

## Hardware notes
- Required: F1 instance, Vivado 2022.x licensed and on PATH
- Run: `./run_pipeline.sh <kernel_dir> <top_module>`
