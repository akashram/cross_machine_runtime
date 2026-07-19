//===- tensor_codec.h - Zero-copy tensor (de)serialization --------------===//
#pragma once

#include <cstddef>
#include <cstdint>
#include <flatbuffers/flatbuffers.h>
#include <vector>

#include "tensor_generated.h" // flatc-generated from tensor.fbs

namespace runtime::fbs_codec {

// Builds a FlatBuffer containing `nbytes` of `data` plus shape/strides/
// dtype/device metadata. Returns an owning, movable buffer — the one copy
// in this path is unavoidable (the caller's tensor storage and the wire
// buffer are different allocations), but it's exactly one copy, not the
// N-copies-through-intermediate-representations serialization libraries
// often incur.
flatbuffers::DetachedBuffer encode_tensor(const void *data, size_t nbytes,
                                           const std::vector<int64_t> &shape,
                                           const std::vector<int64_t> &strides,
                                           runtime::fbs::DType dtype,
                                           runtime::fbs::Device device);

// A view into `buf` — every pointer here aliases into `buf`'s storage.
// `buf` must outlive the view. This is the zero-copy half: unlike
// encode_tensor, decoding never copies the tensor bytes.
struct TensorView {
  const int64_t *shape;
  size_t rank;
  const int64_t *strides;
  runtime::fbs::DType dtype;
  runtime::fbs::Device device;
  const uint8_t *data;
  size_t nbytes;
};

TensorView decode_tensor(const uint8_t *buf, size_t len);

} // namespace runtime::fbs_codec
