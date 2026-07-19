# FlatBuffers Data Plane

**Status: code-complete, not yet built — requires the FlatBuffers compiler (flatc) + library.**

## What this measures
Tensor descriptor serialization without copying — benchmarked against
protobuf (step 6) for hot-path messages.

## Design
`tensor.fbs` mirrors `foundation/`'s unified tensor handle (Phase 1 step
16: device tag, shape, strides, dtype) plus the raw bytes. `encode_tensor`
(`tensor_codec.cpp`) does exactly one copy (tensor storage → wire buffer —
unavoidable, they're different allocations); `decode_tensor` does **zero**
— `TensorView::data` points directly into the caller's buffer. That
asymmetry is the whole reason this step exists separately from step 6's
protobuf control plane: protobuf's generated parser always copies every
`bytes`/`repeated` field into a new owned `std::string`/array on decode,
which is fine for step 6's small, infrequent control messages but wrong
for this step's large, hot-path tensor payloads. `tensor_codec_bench`
measures encode/decode latency at 4KB/1MB/64MB to make that gap visible
once flatc is available to generate `tensor_generated.h`.

## Results
TODO: run on Linux with FlatBuffers installed.

| Size | Encode (µs) | Decode (µs) | Protobuf decode equivalent (µs) |
|------|-------------|-------------|----------------------------------|
| 4 KB | TODO | TODO | TODO |
| 1 MB | TODO | TODO | TODO |
| 64 MB | TODO | TODO | TODO |

## Hardware notes
- Required: flatc + FlatBuffers library (any platform — not Linux-specific,
  just not assumed installed here)
