#include "tensor_codec.h"

#include <stdexcept>

namespace runtime::fbs_codec {

flatbuffers::DetachedBuffer encode_tensor(const void *data, size_t nbytes,
                                           const std::vector<int64_t> &shape,
                                           const std::vector<int64_t> &strides,
                                           runtime::fbs::DType dtype,
                                           runtime::fbs::Device device) {
  flatbuffers::FlatBufferBuilder builder(nbytes + 256); // +256: headroom for the table itself

  auto shapeVec = builder.CreateVector(shape);
  auto stridesVec = builder.CreateVector(strides);
  // CreateVector(ptr, len) copies once, directly into the builder's
  // buffer — no intermediate std::vector<uint8_t> of the tensor bytes.
  auto dataVec = builder.CreateVector(static_cast<const uint8_t *>(data), nbytes);

  auto descriptor = runtime::fbs::CreateTensorDescriptor(
      builder, shapeVec, stridesVec, dtype, device, dataVec);
  builder.Finish(descriptor);

  return builder.Release();
}

TensorView decode_tensor(const uint8_t *buf, size_t len) {
  flatbuffers::Verifier verifier(buf, len);
  if (!runtime::fbs::VerifyTensorDescriptorBuffer(verifier))
    throw std::runtime_error("tensor_codec: malformed TensorDescriptor buffer");

  const runtime::fbs::TensorDescriptor *descriptor = runtime::fbs::GetTensorDescriptor(buf);
  TensorView view;
  view.shape = descriptor->shape()->data();
  view.rank = descriptor->shape()->size();
  view.strides = descriptor->strides()->data();
  view.dtype = descriptor->dtype();
  view.device = descriptor->device();
  view.data = descriptor->data()->data(); // points into `buf` — no copy
  view.nbytes = descriptor->data()->size();
  return view;
}

} // namespace runtime::fbs_codec
