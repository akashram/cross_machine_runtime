// minio_load_gen.cpp — PLAN.md Phase 21 step 11 (MinIO hands-on tuning).
// The portable half of this step: generates real AI-workload-shaped I/O
// artifacts on local disk -- webdataset shards (via data_loading's real
// tar_append/tar_finish, unmodified) and checkpoint shards (via
// checkpoint's real write_shard_sync, unmodified) -- ready to be fed into
// deploy_minio.sh/tune_and_measure.sh's real `mc` commands once MinIO is
// installed (see this directory's README.md: no new local install was
// made this session per the standing policy, so the MinIO-touching half
// stays unrun; this generator is real, complete, and DOES run today).
#include "../../distributed_training/data_loading/webdataset_shard.h"
#include "../../distributed_training/checkpoint/sharded_checkpoint.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace data_loading;
using namespace distributed_training;

namespace {

void generate_webdataset_shards(const std::string &dir, int num_shards, int samples_per_shard,
                                 size_t payload_bytes) {
  fs::create_directories(dir);
  int global_id = 0;
  for (int s = 0; s < num_shards; ++s) {
    std::vector<uint8_t> tar;
    for (int i = 0; i < samples_per_shard; ++i, ++global_id) {
      char key[32];
      std::snprintf(key, sizeof(key), "%08d", global_id);
      std::vector<uint8_t> payload(payload_bytes, static_cast<uint8_t>(global_id & 0xFF));
      tar_append(tar, std::string(key) + ".data", payload);
    }
    tar_finish(tar);
    std::string path = dir + "/shard-" + std::to_string(s) + ".tar";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(tar.data()), static_cast<std::streamsize>(tar.size()));
  }
}

void generate_checkpoint_shards(const std::string &dir, int num_shards, size_t floats_per_shard) {
  fs::create_directories(dir);
  for (int s = 0; s < num_shards; ++s) {
    std::vector<float> shard(floats_per_shard, static_cast<float>(s) + 0.5f);
    write_shard_sync(dir + "/rank" + std::to_string(s) + ".ckpt", shard);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string out_dir = argc > 1 ? argv[1] : "/tmp/hpc_storage_minio_load";
  std::printf("Generating AI-workload-shaped I/O artifacts in %s\n", out_dir.c_str());

  generate_webdataset_shards(out_dir + "/webdataset", /*num_shards=*/16, /*samples_per_shard=*/200,
                              /*payload_bytes=*/8192);
  generate_checkpoint_shards(out_dir + "/checkpoints", /*num_shards=*/8, /*floats_per_shard=*/5'000'000);

  size_t total_bytes = 0;
  for (const auto &entry : fs::recursive_directory_iterator(out_dir)) {
    if (entry.is_regular_file()) total_bytes += fs::file_size(entry.path());
  }
  std::printf("Generated %.1f MB across webdataset/ and checkpoints/ -- ready for deploy_minio.sh + "
              "tune_and_measure.sh once MinIO is installed.\n",
              static_cast<double>(total_bytes) / (1024.0 * 1024.0));
  return total_bytes > 0 ? 0 : 1;
}
