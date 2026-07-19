// tensor_codec_bench.cpp — measures encode_tensor/decode_tensor latency
// at representative tensor sizes, and specifically the *decode* cost gap
// vs. a hypothetical protobuf equivalent (protobuf's generated parse
// always copies every field, including repeated `bytes`, into a new
// std::string — there's no zero-copy decode path). This is what step 7's
// README table is actually arguing: the win isn't encode time, it's that
// decode_tensor's data pointer aliases the wire buffer.

#include "tensor_codec.h"

#include <chrono>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;

int main() {
  std::vector<size_t> sizes = {4096, 1 << 20, 64 << 20}; // 4KB, 1MB, 64MB
  constexpr int kIterations = 100;

  std::printf("%-12s %14s %14s\n", "size", "encode (us)", "decode (us)");
  for (size_t size : sizes) {
    std::vector<uint8_t> data(size, 0x42);
    std::vector<int64_t> shape = {static_cast<int64_t>(size)};
    std::vector<int64_t> strides = {1};

    double encodeUs = 0, decodeUs = 0;
    flatbuffers::DetachedBuffer last;
    for (int i = 0; i < kIterations; ++i) {
      auto t0 = Clock::now();
      auto buf = runtime::fbs_codec::encode_tensor(
          data.data(), data.size(), shape, strides,
          runtime::fbs::DType::F32, runtime::fbs::Device::CPU);
      encodeUs += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

      auto t1 = Clock::now();
      auto view = runtime::fbs_codec::decode_tensor(buf.data(), buf.size());
      decodeUs += std::chrono::duration<double, std::micro>(Clock::now() - t1).count();
      (void)view;
      last = std::move(buf);
    }
    std::printf("%-12zu %14.2f %14.2f\n", size, encodeUs / kIterations, decodeUs / kIterations);
  }
  return 0;
}
