# f1_setup

**Status: code complete — requires AWS F1 instance (Xilinx UltraScale+ VU9P) to run.**

## What this validates
`setup_f1.sh` is the one-time check run right after SSH-ing into a freshly
launched F1 instance, before any other fpga_engine/ step. It proves, in
order:

1. XRT (`xbutil`) is installed and can talk to the driver.
2. The FPGA card is enumerated as a PCIe device.
3. The base shell AFI is loaded into slot 0 (`fpga-describe-local-image`) —
   without this, no user bitstream can be programmed.
4. `vitis_hls` is on PATH and licensed.
5. `vivado` is on PATH and licensed.
6. An actual Vitis "hello world" kernel builds for `TARGET=hw` and runs
   correctly against XRT — the only step here that proves the full chain
   (HLS synthesis → bitstream → XRT execution) works, not just that
   individual binaries exist.

Any failure aborts with a specific, actionable message (missing PATH entry,
unloaded AFI, etc.) rather than a generic error, since this is meant to be
the first thing run on a brand-new instance with no other context yet.

## Results
TODO: run on F1 hardware.

| Check | Result |
|-------|--------|
| xbutil / XRT present | TODO |
| FPGA card enumerated | TODO |
| Shell AFI loaded (slot 0) | TODO |
| vitis_hls present | TODO |
| vivado present | TODO |
| hello_world hw build+run | TODO |

## Hardware notes
- Required: f1.2xlarge or f1.4xlarge, FPGA Developer AMI (Vitis + aws-fpga SDK preinstalled)
- Run: `./setup_f1.sh` (override `F1_PLATFORM` env var if the AMI ships a different platform than `xilinx_aws-vu9p-f1_shell-v04261818_201920_2`)
