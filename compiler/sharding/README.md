# Auto-Sharding Pass (GSPMD-style)

**Status: code-complete, not yet built — requires MLIR on Linux. Multi-GPU validation requires p4d.**

## What this measures
GSPMD-style sharding: annotate tensors with sharding specs (replicated, split on dim N),
propagate specs through ops, insert all-gather/reduce-scatter where sharding changes.

## Design
Applies one of four canned strategies (`ShardingPass.cpp`) uniformly to
every `runtime.matmul`, rather than a full GSPMD cost-based search over
assignments — this validates the mechanics (shard-spec attribute
propagation via `runtime.shard_spec`, collective insertion) that Phase 6's
Megatron-style tensor/data parallelism will actually use:
- **DataParallel**: shard the batch/token dim; weight replicated; no
  collective in the forward matmul (grad all-reduce is a training-loop
  concern, Phase 6).
- **TensorParallelColumn**: shard the weight's output-feature dim; result
  stays column-sharded so it can feed a row-parallel layer without ever
  materializing the full activation (the actual point of Megatron's
  column→row pairing).
- **TensorParallelRow**: shard the contraction dim on both operands, then
  `reduce_scatter` the partial sums — this dialect has no distinct
  "partial-sum" tensor state (unlike real GSPMD/XLA SPMD), so
  reduce_scatter stands in for all-reduce+redistribute. Documented
  simplification.
- **Replicated**: sanity-check baseline, no sharding.

## Results
TODO: run on Linux + multi-GPU hardware.

| Sharding config | Comm ops inserted | Correctness vs unsharded |
|----------------|------------------|--------------------------|
| 4-way data parallel | reduce-scatter + all-gather | TODO |
| 4-way tensor parallel (matmul) | all-reduce | TODO |

## Hardware notes
- Required: MLIR on Linux; multi-GPU for end-to-end validation
