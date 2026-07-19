//===- RuntimeTypes.td - Enum + attribute defs for the runtime dialect --===//
//
// Shared attributes used across ops in the `runtime` dialect: the device
// enum consumed/produced by the placement pass, the reduce-kind enum used
// by ReduceOp, and the shard-spec attribute produced by the auto-sharding
// pass (step 10) and consumed by kernel specialization (step 11).
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_TYPES_TD
#define RUNTIME_TYPES_TD

include "mlir/IR/EnumAttr.td"
include "RuntimeDialect.td"

//===----------------------------------------------------------------------===//
// DeviceKind — written by the placement pass (step 9), read by kernel
// specialization (step 11) and the AOT pipeline (step 12) to pick a
// codegen backend.
//===----------------------------------------------------------------------===//

def Runtime_DeviceKind : I32EnumAttr<"DeviceKind",
    "target execution device for a runtime op", [
      I32EnumAttrCase<"CPU",  0, "cpu">,
      I32EnumAttrCase<"GPU",  1, "gpu">,
      I32EnumAttrCase<"FPGA", 2, "fpga">,
      I32EnumAttrCase<"TPU",  3, "tpu">,
      I32EnumAttrCase<"Unassigned", 4, "unassigned">,
    ]> {
  let cppNamespace = "::runtime";
  let genSpecializedAttr = 0;
}

def Runtime_DeviceAttr : EnumAttr<Runtime_Dialect, Runtime_DeviceKind, "device"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ReduceKind — consumed by ReduceOp (sum/max/mean over a set of axes).
//===----------------------------------------------------------------------===//

def Runtime_ReduceKind : I32EnumAttr<"ReduceKind", "reduction operator", [
      I32EnumAttrCase<"SUM",  0, "sum">,
      I32EnumAttrCase<"MAX",  1, "max">,
      I32EnumAttrCase<"MEAN", 2, "mean">,
    ]> {
  let cppNamespace = "::runtime";
  let genSpecializedAttr = 0;
}

def Runtime_ReduceKindAttr : EnumAttr<Runtime_Dialect, Runtime_ReduceKind, "reduce_kind">;

//===----------------------------------------------------------------------===//
// ShardSpecAttr — per-tensor sharding spec produced by the auto-sharding
// pass (step 10). `partitions[i]` is the number of ways dimension `i` is
// split; -1 means that dimension is replicated (not split) across
// `numDevices`. Length of `partitions` must equal the tensor's rank.
//
// Example: a [1024, 4096] weight matrix column-sharded 4-ways is
// `#runtime.shard_spec<4, [-1, 4]>`.
//===----------------------------------------------------------------------===//

def Runtime_ShardSpecAttr : AttrDef<Runtime_Dialect, "ShardSpec"> {
  let mnemonic = "shard_spec";
  let summary = "per-dimension sharding factors for a tensor value";
  let parameters = (ins
    "int64_t":$numDevices,
    ArrayRefParameter<"int64_t">:$partitions
  );
  let assemblyFormat = "`<` $numDevices `,` `[` $partitions `]` `>`";
  let genVerifyDecl = 1;
}

#endif // RUNTIME_TYPES_TD
