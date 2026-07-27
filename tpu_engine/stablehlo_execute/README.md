# stablehlo_execute

**Status: code-complete, hardware/toolchain-gated — real JAX script, unrun.
No TPU device on this Mac, and no serialized StableHLO artifact to feed it
(producing one needs step 3's MLIR+StableHLO build).**

## What this measures

PLAN.md Phase 8 step 4: execute step 3's lowered StableHLO via JAX, validate
outputs match a CPU/numpy reference.

## Design

- `execute_stablehlo.py` loads a serialized StableHLO artifact via
  `jax.export.deserialize(...).call(*args)` and compares its output against
  a hand-written numpy reference for the same computation, for two of
  `stablehlo_lower/README.md`'s validation rows: `matmul -> bias-add ->
  relu` and `softmax`.
- Deliberately two-stage rather than one script: step 3 produces the
  StableHLO artifact (needs MLIR+StableHLO built from source), step 4
  consumes it (needs only JAX). Keeping them separate means step 4 could in
  principle run against *any* StableHLO producer (e.g. `stablehlo-opt`
  output directly, bypassing the C++ pass) — useful for isolating "is the
  lowering pass wrong" from "is JAX's StableHLO execution path wrong" once
  both pieces exist.
- API note called out in the script's docstring: `jax.export` was
  `jax.experimental.export` before JAX stabilized the name. Confirm the
  exact surface against whatever JAX version `gcp_setup/provision_tpu_vm.sh`
  installs before relying on this unrun.

## Results
TODO: run on a TPU VM once a step-3 StableHLO artifact exists (needs
MLIR+StableHLO built on Linux first, then transferred to the TPU VM, or
built directly on the TPU VM's Linux host).

| Module | max abs err vs numpy |
|--------|----------------------|
| matmul(256x256) → bias-add → relu | TODO |
| softmax(8x512) | TODO |

## Hardware notes
- Required: GCP TPU VM (JAX execution) + a serialized StableHLO artifact
  from step 3's pass (or an equivalent `stablehlo-opt`-produced module, per
  the Design note above).
