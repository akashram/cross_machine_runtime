//===- webdataset_shard.h - Minimal WebDataset tar-shard codec ---------===//
//
// WebDataset's on-disk format is just POSIX tar: a shard is a .tar file,
// and consecutive entries that share a "key" (the filename up to the
// first '.') form one training sample, e.g. `000042.jpg` + `000042.cls`
// are two files of the sample keyed `000042`. The grouping-by-adjacency
// is why WebDataset shards stream well — a reader never needs to seek,
// it just watches for the key to change. This file implements exactly
// that: a USTAR reader/writer good enough to round-trip real shards
// without linking libarchive (its own dependency is just <cstdint>).
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace data_loading {

// One WebDataset sample: files sharing a key, keyed by extension (the
// part of the filename after the first '.'), e.g. {"jpg": <bytes>,
// "cls": <bytes>}.
struct Sample {
  std::string key;
  std::map<std::string, std::vector<uint8_t>> files;
};

// Appends one USTAR entry (header + data, padded to 512-byte blocks) to
// `out`. Test-only helper for synthesizing shards — real shards are
// produced by whatever wrote the dataset (webdataset's own `TarWriter`,
// `tar`, etc.), never by this codebase.
void tar_append(std::vector<uint8_t> &out, const std::string &name,
                 const std::vector<uint8_t> &data);

// Appends the two all-zero 512-byte blocks that mark end-of-archive.
void tar_finish(std::vector<uint8_t> &out);

// Streaming USTAR reader over one shard file. Reads one (name, data) tar
// entry per `next()` call; returns nullopt at end-of-archive or on the
// first malformed header (truncated/corrupt shards are skipped, not
// fatal — one bad shard shouldn't crash a training run).
class TarReader {
public:
  explicit TarReader(const std::string &path);
  ~TarReader();

  TarReader(const TarReader &) = delete;
  TarReader &operator=(const TarReader &) = delete;

  bool is_open() const { return file_ != nullptr; }
  std::optional<std::pair<std::string, std::vector<uint8_t>>> next();

private:
  void *file_; // FILE*, opaque here to keep <cstdio> out of the header
};

// Groups a shard's tar entries into samples by key adjacency (the
// WebDataset contract: entries for one key are contiguous). A key change
// closes the sample being built and opens a new one; end-of-archive
// closes whatever sample is in progress.
class WebDatasetShardReader {
public:
  explicit WebDatasetShardReader(const std::string &shard_path);

  bool is_open() const { return reader_.is_open(); }

  // Returns the next complete sample, or nullopt once the shard is
  // exhausted.
  std::optional<Sample> next();

private:
  TarReader reader_;
  std::optional<Sample> pending_; // entry already read but not yet consumed (key changed)
};

} // namespace data_loading
