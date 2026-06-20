"""
lower_to_stablehlo.py — lower runtime dialect IR to StableHLO for TPU execution
TODO: implement on Linux/Mac with MLIR + StableHLO installed
"""

# TODO: implement on MLIR build + GCP TPU
#
# import mlir.dialects.stablehlo as stablehlo
# from runtime import RuntimeDialect
#
# def lower_runtime_to_stablehlo(module: mlir.ir.Module) -> mlir.ir.Module:
#     """Run lowering pass: runtime.matmul → stablehlo.dot_general, etc."""
#     pm = mlir.passmanager.PassManager.parse("builtin.module(lower-runtime-to-stablehlo)")
#     pm.run(module.operation)
#     return module

print("lower_to_stablehlo: STUB — implement on Linux with MLIR + StableHLO")
