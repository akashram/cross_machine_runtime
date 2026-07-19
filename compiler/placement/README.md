# Device Placement Pass

**Status: code-complete, not yet built — requires MLIR on Linux. Full validation requires CPU+GPU+FPGA.**

## What this measures
Cost-model-driven assignment of each op to CPU/GPU/FPGA/TPU. Minimizes total
cost including compute time and data transfer costs between devices.

## Design
Greedy, not globally optimal (`PlacementPass.cpp`): for each placeable op
(matmul, conv, elementwise, reduce, fusion_group), evaluate
`CostModel::estimate_us` on all four devices, add the transfer-in cost for
any operand whose producer landed elsewhere (function arguments default to
CPU-resident), and pick the minimum. The chosen device is written both as a
`runtime.device` attribute (for kernel specialization, step 11, to read)
and made physically explicit by inserting a `runtime.transfer` op at every
device-boundary-crossing operand — nothing downstream has to re-derive
placement from attributes alone. A real DP/ILP scheduler that considers how
op N's placement constrains op N+1's transfer cost is noted as future work;
greedy is the right starting point before there's a real workload DAG to
validate the alternative against.

## Results
TODO: run on Linux with MLIR and validate against multi-device instance.

| Op | Cost model choice | Reason |
|----|------------------|--------|
| matmul (large) | GPU | TODO |
| embedding lookup | CPU | TODO |
| conv (small) | FPGA | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built; multi-device for validation
