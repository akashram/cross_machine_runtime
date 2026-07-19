//===- RuntimeOps.td - Op definitions for the runtime dialect -----------===//
//
// Step 2-3 of Phase 4: op set for the tensor-op IR that every later pass
// (shape inference, fusion, memory planning, remat, placement, sharding,
// kernel specialization) operates on.
//
// Design choices, see compiler/DESIGN.md for the full writeup:
//  - Operands/results are builtin `RankedTensorType` (dynamic dims via `?`),
//    not a custom tensor type. Device placement and sharding are carried as
//    *discardable attributes* on the op (`runtime.device`, `runtime.shard`)
//    rather than baked into the type — this keeps every pass before
//    placement free to reason about pure dataflow, and lets placement/
//    sharding be re-run without changing every op's type signature.
//  - `Runtime_FusionGroupOp` is a single-region op (like `scf.if`) so the
//    fusion pass (step 5) can move a matched subgraph into one op without
//    inventing a new op per fusable pattern.
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_OPS_TD
#define RUNTIME_OPS_TD

include "RuntimeDialect.td"
include "RuntimeTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class Runtime_Op<string mnemonic, list<Trait> traits = []>
    : Op<Runtime_Dialect, mnemonic, traits>;

// Discardable attribute names shared by every op below (set post-hoc by
// placement/sharding, not part of the op's inherent attribute list, so
// ops keep working before those passes run).
defvar RuntimeDeviceAttrName = "runtime.device";
defvar RuntimeShardAttrName  = "runtime.shard";

def AnyRuntimeTensor : TensorOf<[AnyFloat, AnyInteger]>;

//===----------------------------------------------------------------------===//
// Compute ops
//===----------------------------------------------------------------------===//

def Runtime_MatmulOp : Runtime_Op<"matmul", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "batched matrix multiply, with optional fused bias add";
  let arguments = (ins
    AnyRuntimeTensor:$lhs,
    AnyRuntimeTensor:$rhs,
    Optional<AnyRuntimeTensor>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`,` `bias` `=` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Runtime_ConvOp : Runtime_Op<"conv", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "NCHW convolution with optional bias";
  let arguments = (ins
    AnyRuntimeTensor:$input,
    AnyRuntimeTensor:$filter,
    Optional<AnyRuntimeTensor>:$bias,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilations
  );
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = [{
    $input `,` $filter (`,` `bias` `=` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

class Runtime_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []>
    : Runtime_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType,
        DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
  let arguments = (ins AnyRuntimeTensor:$input);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Runtime_ReluOp    : Runtime_ElementwiseUnaryOp<"relu">;
def Runtime_GeluOp    : Runtime_ElementwiseUnaryOp<"gelu">;
def Runtime_SigmoidOp : Runtime_ElementwiseUnaryOp<"sigmoid">;

class Runtime_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []>
    : Runtime_Op<mnemonic, !listconcat(traits, [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
  let arguments = (ins AnyRuntimeTensor:$lhs, AnyRuntimeTensor:$rhs);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Runtime_AddOp : Runtime_ElementwiseBinaryOp<"add", [Commutative]>;
def Runtime_MulOp : Runtime_ElementwiseBinaryOp<"mul", [Commutative]>;
def Runtime_SubOp : Runtime_ElementwiseBinaryOp<"sub", []>;

def Runtime_SoftmaxOp : Runtime_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "softmax along a single axis";
  let arguments = (ins AnyRuntimeTensor:$input, I64Attr:$axis);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input `axis` `=` $axis attr-dict `:` type($input)";
}

def Runtime_ReduceOp : Runtime_Op<"reduce", [Pure]> {
  let summary = "reduce (sum/max/mean) over a set of axes";
  let arguments = (ins
    AnyRuntimeTensor:$input,
    I64ArrayAttr:$axes,
    Runtime_ReduceKindAttr:$kind,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$kind $input `axes` `=` $axes attr-dict `:` functional-type(operands, results)";
}

def Runtime_GatherOp : Runtime_Op<"gather", [Pure]> {
  let summary = "gather rows of `input` at `indices` along `axis`";
  let arguments = (ins AnyRuntimeTensor:$input, AnyRuntimeTensor:$indices, I64Attr:$axis);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input `,` $indices `axis` `=` $axis attr-dict `:` functional-type(operands, results)";
}

def Runtime_ScatterOp : Runtime_Op<"scatter", [Pure]> {
  let summary = "scatter `updates` into `input` at `indices` along `axis`";
  let arguments = (ins
    AnyRuntimeTensor:$input, AnyRuntimeTensor:$indices,
    AnyRuntimeTensor:$updates, I64Attr:$axis
  );
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input `,` $indices `,` $updates `axis` `=` $axis attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Ops inserted by later passes (not present in user-authored IR)
//===----------------------------------------------------------------------===//

def Runtime_YieldOp : Runtime_Op<"yield", [Pure, Terminator,
    ParentOneOf<["FusionGroupOp"]>]> {
  let summary = "terminator for runtime.fusion_group's body region";
  let arguments = (ins Variadic<AnyRuntimeTensor>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";
}

def Runtime_FusionGroupOp : Runtime_Op<"fusion_group", [
    SingleBlock, NoTerminator<"never">,
    RecursiveMemoryEffects, IsolatedFromAbove]> {
  let summary = "opaque group of fused ops, produced by the fusion pass (step 5)";
  let description = [{
    Holds a matched subgraph (e.g. matmul -> add(bias) -> relu) in a single
    region so downstream passes (placement, kernel specialization) treat the
    whole group as one scheduling/codegen unit. `fusion_kind` names the
    matched pattern for kernel specialization's lookup table (step 11).
  }];
  let arguments = (ins Variadic<AnyRuntimeTensor>:$inputs, StrAttr:$fusion_kind);
  let results = (outs Variadic<AnyRuntimeTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $fusion_kind `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)
    $body
  }];
}

def Runtime_TransferOp : Runtime_Op<"transfer", [Pure, SameOperandsAndResultType]> {
  let summary = "explicit cross-device data movement, inserted by placement (step 9)";
  let arguments = (ins AnyRuntimeTensor:$input, Runtime_DeviceAttr:$src, Runtime_DeviceAttr:$dst);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input $src `->` $dst attr-dict `:` type($input)";
}

class Runtime_CollectiveOp<string mnemonic>
    : Runtime_Op<mnemonic, [SameOperandsAndResultType]> {
  let arguments = (ins AnyRuntimeTensor:$input, I64Attr:$shard_dim, I64Attr:$num_shards);
  let results = (outs AnyRuntimeTensor:$result);
  let assemblyFormat = "$input `dim` `=` $shard_dim `shards` `=` $num_shards attr-dict `:` type($input)";
}

def Runtime_AllGatherOp     : Runtime_CollectiveOp<"all_gather">;
def Runtime_ReduceScatterOp : Runtime_CollectiveOp<"reduce_scatter">;

def Runtime_KernelCallOp : Runtime_Op<"kernel_call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call into a device-specialized kernel, produced by step 11";
  let description = [{
    Terminal op for the `runtime` dialect: after kernel specialization, a
    `runtime.fusion_group` (or standalone compute op) placed on a device
    lowers to `runtime.kernel_call` referencing a symbol implemented in
    cpu_engine/, gpu_engine/, or fpga_engine/. The AOT pipeline (step 12)
    resolves `callee` to an actual object-file symbol at link time.
  }];
  let arguments = (ins
    Variadic<AnyRuntimeTensor>:$inputs,
    FlatSymbolRefAttr:$callee,
    Runtime_DeviceAttr:$device
  );
  let results = (outs Variadic<AnyRuntimeTensor>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)` `on` $device attr-dict `:` functional-type($inputs, $results)
  }];
}

#endif // RUNTIME_OPS_TD
