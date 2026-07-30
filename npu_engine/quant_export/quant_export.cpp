#include "quant_export.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace npu_engine {

namespace {

void put_u32(std::vector<uint8_t> &buf, uint32_t v) {
  uint8_t bytes[4];
  std::memcpy(bytes, &v, 4);
  buf.insert(buf.end(), bytes, bytes + 4);
}
void put_i32(std::vector<uint8_t> &buf, int32_t v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  put_u32(buf, u);
}
void put_f32(std::vector<uint8_t> &buf, float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  put_u32(buf, u);
}

uint32_t get_u32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
int32_t get_i32(const uint8_t *p) {
  int32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
float get_f32(const uint8_t *p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

} // namespace

std::vector<uint8_t> serialize_npu_weight(const QuantizedWeight &q) {
  std::vector<uint8_t> buf;
  buf.reserve(16 + q.qweight.size() + q.scales.size() * 4 + q.zeros.size() * 4);

  put_u32(buf, kNpuExportMagic);
  put_u32(buf, kNpuExportVersion);
  put_i32(buf, q.rows);
  put_i32(buf, q.cols);
  put_i32(buf, q.group_size);
  put_i32(buf, q.bits);

  if (q.bits != 8) {
    throw std::runtime_error("serialize_npu_weight: this export format assumes bits=8 "
                              "(one byte per weight, no sub-byte packing) — got a different bit width");
  }

  for (int32_t v : q.qweight) {
    if (v < 0 || v > 255)
      throw std::runtime_error("serialize_npu_weight: quantized value out of uint8 range for bits=8");
    buf.push_back(static_cast<uint8_t>(v));
  }
  for (float s : q.scales) put_f32(buf, s);
  for (int32_t z : q.zeros) put_i32(buf, z);

  return buf;
}

QuantizedWeight deserialize_npu_weight(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < 24) throw std::runtime_error("deserialize_npu_weight: buffer too small for header");

  const uint8_t *p = bytes.data();
  uint32_t magic = get_u32(p); p += 4;
  uint32_t version = get_u32(p); p += 4;
  if (magic != kNpuExportMagic) throw std::runtime_error("deserialize_npu_weight: bad magic");
  if (version != kNpuExportVersion) throw std::runtime_error("deserialize_npu_weight: unsupported version");

  QuantizedWeight q;
  q.rows = get_i32(p); p += 4;
  q.cols = get_i32(p); p += 4;
  q.group_size = get_i32(p); p += 4;
  q.bits = get_i32(p); p += 4;

  size_t num_weights = static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols);
  int num_groups = q.num_groups();
  size_t num_scale_entries = static_cast<size_t>(q.rows) * static_cast<size_t>(num_groups);

  size_t expected = 24 + num_weights + num_scale_entries * 4 + num_scale_entries * 4;
  if (bytes.size() != expected)
    throw std::runtime_error("deserialize_npu_weight: buffer size does not match header-derived layout");

  q.qweight.resize(num_weights);
  for (size_t i = 0; i < num_weights; ++i) q.qweight[i] = static_cast<int32_t>(p[i]);
  p += num_weights;

  q.scales.resize(num_scale_entries);
  for (size_t i = 0; i < num_scale_entries; ++i) { q.scales[i] = get_f32(p); p += 4; }

  q.zeros.resize(num_scale_entries);
  for (size_t i = 0; i < num_scale_entries; ++i) { q.zeros[i] = get_i32(p); p += 4; }

  return q;
}

void write_npu_weight_file(const std::string &path, const QuantizedWeight &q) {
  std::vector<uint8_t> buf = serialize_npu_weight(q);
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("write_npu_weight_file: could not open " + path);
  out.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

QuantizedWeight read_npu_weight_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("read_npu_weight_file: could not open " + path);
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  in.read(reinterpret_cast<char *>(buf.data()), size);
  return deserialize_npu_weight(buf);
}

size_t fp32_byte_size(const QuantizedWeight &q) {
  return static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols) * 4;
}

size_t npu_export_byte_size(const QuantizedWeight &q) {
  int num_groups = q.num_groups();
  size_t num_weights = static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols);
  size_t num_scale_entries = static_cast<size_t>(q.rows) * static_cast<size_t>(num_groups);
  return 24 + num_weights + num_scale_entries * 4 + num_scale_entries * 4;
}

} // namespace npu_engine
