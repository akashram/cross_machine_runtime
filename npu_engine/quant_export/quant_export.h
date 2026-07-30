#pragma once
#include "../../inference_serving/gptq/gptq.h"

#include <cstdint>
#include <string>
#include <vector>

// PLAN.md Phase 15 step 2: quantization export pipeline. Reuses
// inference_serving::GptqQuantizer directly (Hessian-guided greedy
// column quantization, Frantar et al. 2022) rather than reimplementing
// quantization math — the only new code here is the NPU-facing part: a
// serialization format for the quantized weight, since neither `onnx`
// nor `coremltools` is installed locally (no numpy/onnx/protoc on this
// Mac either — see npu_engine/README.md's platform notes). Vitis AI's
// flow (fpga_engine/vitis_ai) hit the same "quantizer needs a framework
// format" wall for a different accelerator; this step hits it for NPU
// export specifically.
//
// bits=8 (INT8) is used here, not GPTQ's usual bits=4 default: NPU
// toolchains (CoreML, ONNX Runtime NPU execution providers) quantize to
// INT8 as their default/most portable path — INT4 is common for LLM
// *weight-only* GPU serving (inference_serving/gptq's own use case) but
// isn't the NPU-typical target, so this step deliberately reuses the
// same GptqQuantizer class with a different bit width rather than a
// separate implementation.

namespace npu_engine {

using inference_serving::GptqQuantizer;
using inference_serving::QuantizedWeight;

// Custom binary container for a single quantized weight matrix, since no
// real ONNX/CoreML writer is available locally. bits=8 means qweight
// values fit exactly one byte each (0..255) with no sub-byte packing
// needed -- unlike gptq.h's bits=4 case, byte alignment falls out for
// free here, so this format stores one uint8 per weight, one float32
// scale + one int32 zero-point per (row, group).
//
// Layout (little-endian, matches this machine's endianness -- no
// cross-endian portability claim made or needed for a same-machine
// round-trip test):
//   uint32_t magic       (kNpuExportMagic)
//   uint32_t version     (kNpuExportVersion)
//   int32_t  rows, cols, group_size, bits
//   uint8_t  qweight[rows*cols]              -- row-major
//   float    scales[rows*num_groups]
//   int32_t  zeros[rows*num_groups]
constexpr uint32_t kNpuExportMagic = 0x4E505551; // arbitrary magic constant ("NPUQ"-derived)
constexpr uint32_t kNpuExportVersion = 1;

std::vector<uint8_t> serialize_npu_weight(const QuantizedWeight &q);
QuantizedWeight deserialize_npu_weight(const std::vector<uint8_t> &bytes);

void write_npu_weight_file(const std::string &path, const QuantizedWeight &q);
QuantizedWeight read_npu_weight_file(const std::string &path);

// Byte count of the dense FP32 original vs. this export format, for a
// real (not assumed) compression ratio measurement.
size_t fp32_byte_size(const QuantizedWeight &q);
size_t npu_export_byte_size(const QuantizedWeight &q);

} // namespace npu_engine
