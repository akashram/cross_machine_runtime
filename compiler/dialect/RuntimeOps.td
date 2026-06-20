// RuntimeOps.td — TableGen definitions for the runtime dialect
// TODO: implement on Linux with MLIR

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// TODO: define DeviceAttr (CPU, GPU, FPGA, TPU) and DTypeAttr

def Runtime_Dialect : Dialect {
  let name = "runtime";
  let summary = "Cross-machine runtime ops for heterogeneous ML execution";
  let cppNamespace = "::runtime";
}

class Runtime_Op<string mnemonic, list<Trait> traits = []>
    : Op<Runtime_Dialect, mnemonic, traits>;

// TODO: define ops
// def Runtime_MatmulOp : Runtime_Op<"matmul"> { ... }
// def Runtime_ConvOp   : Runtime_Op<"conv">   { ... }
// def Runtime_ReluOp   : Runtime_Op<"relu", [Pure]> { ... }
// def Runtime_AddOp    : Runtime_Op<"add",  [Pure, Commutative]> { ... }
// def Runtime_SoftmaxOp: Runtime_Op<"softmax"> { ... }
// def Runtime_ReduceOp : Runtime_Op<"reduce"> { ... }
