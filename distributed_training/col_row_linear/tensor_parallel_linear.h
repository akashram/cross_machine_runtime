#pragma once

// Column/row-parallel linear layers (Megatron-LM style tensor parallelism):
// PLAN.md step 11. Written as hand-derived forward/backward Matrix
// functions rather than new autograd.h Tensor ops — real ML systems do
// exactly this for fused tensor-parallel blocks too (a custom autograd
// Function, not a composition of primitive ops), and it keeps step 6's
// generic Tensor/Node tape from needing a Channel-aware op type for what
// is really one specific fused pattern.
//
// The classic Megatron MLP block this validates: ColumnParallelLinear
// (splits the weight by OUTPUT columns — forward needs no communication,
// since the input is already replicated) -> activation (elementwise, still
// no communication) -> RowParallelLinear (splits the weight by INPUT rows
// — forward needs one all-reduce SUM to combine each rank's partial
// output). Chaining column-then-row this way needs exactly ONE
// communication op in the forward pass (the row-parallel's all-reduce) and
// ONE in the backward pass (the column-parallel's all-reduce for dX) —
// not one after every layer, which is the entire efficiency point of this
// specific pairing.

#include "matrix.h"

namespace distributed_training {

// Weight sharded by OUTPUT columns: this rank owns W[:, shard]. Forward
// needs no communication (X is replicated). Backward's dX contribution is
// only PARTIAL (this rank only sees dY for its own output columns) and
// must be summed across tensor-parallel ranks by the caller.
struct ColumnParallelLinear {
  Matrix weight_shard; // [in x out_shard]
  Matrix bias_shard;   // [1 x out_shard]

  Matrix forward(const Matrix &x) const { return x.matmul(weight_shard).add_row_broadcast(bias_shard); }

  struct Grads {
    Matrix dweight;   // [in x out_shard], local
    Matrix dbias;     // [1 x out_shard], local
    Matrix dx_partial; // [batch x in], PARTIAL — caller must all-reduce (sum) across TP ranks
  };

  Grads backward(const Matrix &x, const Matrix &dy_shard) const {
    return Grads{x.transpose().matmul(dy_shard), dy_shard.sum_rows(), dy_shard.matmul(weight_shard.transpose())};
  }
};

// Weight sharded by INPUT rows: this rank owns W[shard, :], and expects an
// already-sharded input x_shard (this TP rank's column-slice of the
// upstream activation). Forward produces only a PARTIAL output sum, which
// the caller must all-reduce (sum) to get the true output — after which
// gradient w.r.t. that output is already correct and IDENTICAL on every
// rank, so backward needs no further communication (see the module
// comment above for why exactly one all-reduce lands in each direction).
// Bias is intentionally NOT part of this struct: Megatron replicates the
// row-parallel layer's bias (added once, after the all-reduce) rather
// than sharding it — sharding a bias saves negligible memory (O(out), not
// O(in*out)) for the cost of another communication op.
struct RowParallelLinear {
  Matrix weight_shard; // [in_shard x out]

  Matrix forward(const Matrix &x_shard) const { return x_shard.matmul(weight_shard); }

  struct Grads {
    Matrix dweight; // [in_shard x out], local
    Matrix dx_shard; // [batch x in_shard], local (no communication needed — see above)
  };

  // dy_full: gradient w.r.t. this layer's OUTPUT, already the correct,
  // fully-reduced value (identical on every rank, since the forward
  // all-reduce already made the output itself identical everywhere).
  Grads backward(const Matrix &x_shard, const Matrix &dy_full) const {
    return Grads{x_shard.transpose().matmul(dy_full), dy_full.matmul(weight_shard.transpose())};
  }
};

} // namespace distributed_training
