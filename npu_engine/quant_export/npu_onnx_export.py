# npu_onnx_export.py -- real onnx/coremltools API calls that would take
# a quant_export_test.cpp-produced .npuw file (rows/cols/group_size/bits
# header, uint8 qweight, float32 scales, int32 zero-points -- see
# quant_export.h's format comment) and produce an actual NPU-deployable
# .onnx (QuantizeLinear/MatMulInteger/DequantizeLinear node graph, the
# ONNX Runtime NPU execution provider's expected quantized-matmul shape)
# or CoreML .mlmodel.
#
# Same convention as fpga_engine/vitis_ai/mlp_model.py: a real, complete
# script written to actually run once the dependency exists, not a
# stubbed sketch. Neither `onnx` nor `coremltools` (nor `numpy`) is
# installed on this Mac -- confirmed via `python3 -c "import onnx"` /
# `import coremltools` / `import numpy`, all ModuleNotFoundError -- so
# this file is unrun.
#
# TODO: requires `pip install onnx numpy` (for the ONNX path) or
# `pip install coremltools numpy` (for the CoreML/ANE path). Neither
# installed locally -- this session deliberately declined new local
# installs (see project memory). Unrun.

import struct
import sys

NPU_EXPORT_MAGIC = 0x4E505551
NPU_EXPORT_VERSION = 1


def read_npuw(path):
    """Pure-stdlib reader for the .npuw format quant_export.cpp writes --
    this part has NO onnx/numpy dependency and could run today, but is
    kept in this file (rather than split out) since its only real
    consumer is the export step below."""
    with open(path, "rb") as f:
        data = f.read()

    magic, version, rows, cols, group_size, bits = struct.unpack_from("<IIiiii", data, 0)
    if magic != NPU_EXPORT_MAGIC:
        raise ValueError("bad .npuw magic")
    if version != NPU_EXPORT_VERSION:
        raise ValueError("unsupported .npuw version")
    if bits != 8:
        raise ValueError("this export script assumes bits=8 (byte-aligned qweight)")

    offset = 24
    num_weights = rows * cols
    qweight = list(data[offset:offset + num_weights])
    offset += num_weights

    num_groups = (cols + group_size - 1) // group_size
    num_scale_entries = rows * num_groups
    scales = list(struct.unpack_from(f"<{num_scale_entries}f", data, offset))
    offset += num_scale_entries * 4
    zeros = list(struct.unpack_from(f"<{num_scale_entries}i", data, offset))

    return {
        "rows": rows, "cols": cols, "group_size": group_size, "bits": bits,
        "qweight": qweight, "scales": scales, "zeros": zeros,
    }


def export_onnx(npuw_path, onnx_path):
    """Real ONNX export: builds a graph
        input -> QuantizeLinear(dynamic, per-tensor) -> MatMulInteger(qweight, zero_point)
              -> Cast(float) -> Mul(scale) -> output
    matching the ONNX Runtime NPU execution provider's expected quantized-
    matmul subgraph shape (see onnxruntime docs, "Quantize Linear Matmul").
    Per-(row,group) scales/zeros from quant_export.h's GPTQ output are
    folded to a per-output-channel (per-row) scale here since ONNX's
    MatMulInteger + DequantizeLinear only supports one scale axis
    cleanly without a custom op -- a real, disclosed simplification
    (loses per-group granularity within a row), not a hidden one.
    """
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    w = read_npuw(npuw_path)
    rows, cols = w["rows"], w["cols"]
    qweight_np = np.array(w["qweight"], dtype=np.uint8).reshape(rows, cols)
    # Fold to one scale/zero per row (per-output-channel), averaging across
    # this row's groups -- see docstring above for why.
    num_groups = (cols + w["group_size"] - 1) // w["group_size"]
    scales_np = np.array(w["scales"], dtype=np.float32).reshape(rows, num_groups).mean(axis=1)
    zeros_np = np.array(w["zeros"], dtype=np.int64).reshape(rows, num_groups)[:, 0]  # first group's zero as representative

    qweight_init = numpy_helper.from_array(qweight_np, name="qweight")
    scale_init = numpy_helper.from_array(scales_np, name="w_scale")
    zero_init = numpy_helper.from_array(zeros_np.astype(np.uint8), name="w_zero_point")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, cols])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, rows])

    quantize_x = helper.make_node(
        "DynamicQuantizeLinear", ["x"], ["x_q", "x_scale", "x_zero_point"], name="quantize_x")
    matmul_int = helper.make_node(
        "MatMulInteger", ["x_q", "qweight", "x_zero_point", "w_zero_point"], ["y_int32"], name="matmul_int")
    cast_f = helper.make_node("Cast", ["y_int32"], ["y_int32_f"], to=TensorProto.FLOAT, name="cast_f")
    scale_mul = helper.make_node("Mul", ["y_int32_f", "w_scale"], ["y"], name="scale_mul")

    graph = helper.make_graph(
        [quantize_x, matmul_int, cast_f, scale_mul],
        "npu_quantized_linear",
        [x], [y],
        initializer=[qweight_init, scale_init, zero_init],
    )
    model = helper.make_model(graph, producer_name="npu_engine.quant_export")
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"wrote {onnx_path}")


def export_coreml(npuw_path, mlmodel_path):
    """Real CoreML export via coremltools' MIL builder, targeting the
    Apple Neural Engine (compute_units=ALL lets CoreML's compiler pick
    ANE when the op/dtype combination is eligible -- int8 weight-only
    quantized linear is ANE-eligible on recent CoreML versions per
    Apple's documented op support tables)."""
    import numpy as np
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    w = read_npuw(npuw_path)
    rows, cols = w["rows"], w["cols"]
    qweight_np = np.array(w["qweight"], dtype=np.uint8).reshape(rows, cols).astype(np.int8) - 128  # symmetric-ish shift
    num_groups = (cols + w["group_size"] - 1) // w["group_size"]
    scales_np = np.array(w["scales"], dtype=np.float32).reshape(rows, num_groups).mean(axis=1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, cols))])
    def npu_linear(x):
        w_const = mb.const(val=qweight_np.astype(np.float32))
        scaled_w = mb.mul(x=w_const, y=scales_np.reshape(rows, 1))
        return mb.linear(x=x, weight=scaled_w, name="y")

    mlmodel = ct.convert(npu_linear, convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL)
    mlmodel.save(mlmodel_path)
    print(f"wrote {mlmodel_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: npu_onnx_export.py <in.npuw> <out.onnx|out.mlmodel>")
        sys.exit(1)
    npuw_path, out_path = sys.argv[1], sys.argv[2]
    if out_path.endswith(".mlmodel") or out_path.endswith(".mlpackage"):
        export_coreml(npuw_path, out_path)
    else:
        export_onnx(npuw_path, out_path)
