//===- RuntimeDialect.td - Dialect definition, no ops/types -------------===//
//
// Split out from RuntimeOps.td so that RuntimeTypes.td (attributes) and
// RuntimeOps.td (ops) can both depend on the dialect definition without
// a circular include.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_DIALECT_TD
#define RUNTIME_DIALECT_TD

include "mlir/IR/OpBase.td"

def Runtime_Dialect : Dialect {
  let name = "runtime";
  let cppNamespace = "::runtime";
  let summary = "Cross-machine runtime ops for heterogeneous ML execution";
  let description = [{
    The `runtime` dialect is the single IR that flows through every compiler
    pass in this project: shape inference, fusion, memory planning,
    rematerialization, device placement, auto-sharding, and kernel
    specialization all operate on `runtime` ops. Ops are device-agnostic —
    `runtime.matmul` says nothing about *where* the matmul runs. Placement
    (step 9) attaches a `runtime.device` attribute; kernel specialization
    (step 11) is the only pass that knows this dialect ends its life as
    AVX-512 intrinsics, a CUDA kernel launch, an HLS core, or an XLA HLO op.
  }];

  // Attributes (DeviceAttr, ReduceKindAttr, ShardSpecAttr) print/parse via
  // the default (de)serializer generated from their TableGen definitions —
  // no hand-written printer/parser needed for the common case.
  let useDefaultAttributePrinterParser = 1;
}

#endif // RUNTIME_DIALECT_TD
