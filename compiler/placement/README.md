# Device Placement Pass

**Status: STUB — requires MLIR on Linux. Full validation requires CPU+GPU+FPGA.**

## What this measures
Cost-model-driven assignment of each op to CPU/GPU/FPGA/TPU. Minimizes total
cost including compute time and data transfer costs between devices.

## Results
TODO: run on Linux with MLIR and validate against multi-device instance.

| Op | Cost model choice | Reason |
|----|------------------|--------|
| matmul (large) | GPU | TODO |
| embedding lookup | CPU | TODO |
| conv (small) | FPGA | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built; multi-device for validation
