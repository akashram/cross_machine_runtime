#include "webdataset_shard.h"

#include <cstdio>
#include <cstring>

namespace data_loading {

namespace {

constexpr size_t kBlockSize = 512;

// Writes `value` as a zero-padded, NUL-terminated octal field of width
// `width` (USTAR numeric fields are ASCII octal, not binary).
void write_octal(char *field, size_t width, uint64_t value) {
  std::snprintf(field, width, "%0*llo", static_cast<int>(width - 1),
                static_cast<unsigned long long>(value));
}

uint64_t parse_octal(const char *field, size_t width) {
  uint64_t value = 0;
  for (size_t i = 0; i < width && field[i] >= '0' && field[i] <= '7'; ++i) {
    value = value * 8 + static_cast<uint64_t>(field[i] - '0');
  }
  return value;
}

// USTAR checksum: sum of all header bytes with the checksum field itself
// treated as eight ASCII spaces.
unsigned compute_checksum(const uint8_t header[kBlockSize]) {
  unsigned sum = 0;
  for (size_t i = 0; i < kBlockSize; ++i) {
    bool in_chksum_field = i >= 148 && i < 156;
    sum += in_chksum_field ? ' ' : header[i];
  }
  return sum;
}

// Splits a WebDataset entry name into (key, extension): the key is
// everything before the first '.', the extension is everything after.
// `a/000042.jpg` -> key "a/000042", ext "jpg".
std::pair<std::string, std::string> split_key_ext(const std::string &name) {
  auto slash = name.find_last_of('/');
  auto dot = name.find('.', slash == std::string::npos ? 0 : slash + 1);
  if (dot == std::string::npos) return {name, ""};
  return {name.substr(0, dot), name.substr(dot + 1)};
}

} // namespace

void tar_append(std::vector<uint8_t> &out, const std::string &name,
                 const std::vector<uint8_t> &data) {
  uint8_t header[kBlockSize] = {};
  std::snprintf(reinterpret_cast<char *>(header), 100, "%s", name.c_str());
  write_octal(reinterpret_cast<char *>(header + 100), 8, 0644);
  write_octal(reinterpret_cast<char *>(header + 108), 8, 0);   // uid
  write_octal(reinterpret_cast<char *>(header + 116), 8, 0);   // gid
  write_octal(reinterpret_cast<char *>(header + 124), 12, data.size());
  write_octal(reinterpret_cast<char *>(header + 136), 12, 0);  // mtime
  std::memset(header + 148, ' ', 8);                           // chksum placeholder
  header[156] = '0';                                           // typeflag: regular file
  std::memcpy(header + 257, "ustar", 5);
  header[262] = '\0';
  header[263] = '0';
  header[264] = '0';

  unsigned chksum = compute_checksum(header);
  write_octal(reinterpret_cast<char *>(header + 148), 7, chksum);
  header[155] = '\0';

  out.insert(out.end(), header, header + kBlockSize);
  out.insert(out.end(), data.begin(), data.end());
  size_t padding = (kBlockSize - (data.size() % kBlockSize)) % kBlockSize;
  out.insert(out.end(), padding, 0);
}

void tar_finish(std::vector<uint8_t> &out) {
  out.insert(out.end(), 2 * kBlockSize, 0);
}

TarReader::TarReader(const std::string &path) {
  file_ = std::fopen(path.c_str(), "rb");
}

TarReader::~TarReader() {
  if (file_ != nullptr) std::fclose(static_cast<FILE *>(file_));
}

std::optional<std::pair<std::string, std::vector<uint8_t>>> TarReader::next() {
  if (file_ == nullptr) return std::nullopt;
  FILE *f = static_cast<FILE *>(file_);

  uint8_t header[kBlockSize];
  if (std::fread(header, 1, kBlockSize, f) != kBlockSize) return std::nullopt;

  // End-of-archive: an all-zero header block.
  bool all_zero = true;
  for (uint8_t b : header) {
    if (b != 0) { all_zero = false; break; }
  }
  if (all_zero) return std::nullopt;

  if (std::memcmp(header + 257, "ustar", 5) != 0) return std::nullopt; // not USTAR — treat as EOF

  char typeflag = static_cast<char>(header[156]);
  uint64_t size = parse_octal(reinterpret_cast<char *>(header + 124), 12);

  std::string prefix(reinterpret_cast<char *>(header + 345));
  std::string name(reinterpret_cast<char *>(header));
  std::string full_name = prefix.empty() ? name : prefix + "/" + name;

  std::vector<uint8_t> data(size);
  if (size > 0 && std::fread(data.data(), 1, size, f) != size) return std::nullopt;
  size_t padding = (kBlockSize - (size % kBlockSize)) % kBlockSize;
  if (padding > 0) std::fseek(f, static_cast<long>(padding), SEEK_CUR);

  if (typeflag != '0' && typeflag != '\0') return next(); // skip non-regular entries (dirs, links, ...)

  return std::make_pair(full_name, std::move(data));
}

WebDatasetShardReader::WebDatasetShardReader(const std::string &shard_path)
    : reader_(shard_path) {}

std::optional<Sample> WebDatasetShardReader::next() {
  Sample sample;
  bool have_key = false;

  if (pending_.has_value()) {
    sample = std::move(*pending_);
    pending_.reset();
    have_key = true;
  }

  while (true) {
    auto entry = reader_.next();
    if (!entry.has_value()) {
      return have_key ? std::optional<Sample>(std::move(sample)) : std::nullopt;
    }
    auto [key, ext] = split_key_ext(entry->first);
    if (!have_key) {
      sample.key = key;
      have_key = true;
    } else if (key != sample.key) {
      pending_ = Sample{key, {}};
      pending_->files[ext] = std::move(entry->second);
      return sample;
    }
    sample.files[ext] = std::move(entry->second);
  }
}

} // namespace data_loading
