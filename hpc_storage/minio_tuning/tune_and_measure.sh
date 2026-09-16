#!/usr/bin/env bash
# tune_and_measure.sh — PLAN.md Phase 21 step 11. Real `mc` commands that
# sweep two real, actually-turnable MinIO knobs against the AI-workload-
# shaped data minio_load_gen.cpp produces, timing each configuration with
# real wall-clock. Unrun -- same status as deploy_minio.sh (see this
# directory's README.md).
set -euo pipefail

MINIO_ALIAS="hpcstorage"
LOAD_DIR="${1:-/tmp/hpc_storage_minio_load}"
RESULTS="${2:-/tmp/hpc_storage_minio_tuning_results.txt}"

echo "== Knob 1: part/chunk size sweep (mc cp --part-size) ==" | tee "${RESULTS}"
# MinIO's multipart upload part size directly trades off per-part HTTP
# overhead (many small parts = more round trips) against retry-on-failure
# granularity (large parts = a failed part re-uploads more data). AI
# checkpoint shards (large, sequential -- see io_pattern_characterization's
# real finding) are exactly the workload this knob matters for.
for part_size in 16MiB 64MiB 128MiB; do
  echo "-- part-size=${part_size} --" | tee -a "${RESULTS}"
  /usr/bin/time -p mc cp --part-size "${part_size}" -r "${LOAD_DIR}/checkpoints" \
    "${MINIO_ALIAS}/hpc-tuning-bucket/checkpoints-${part_size}" 2>&1 | tee -a "${RESULTS}"
done

echo "== Knob 2: concurrent-connection limit (parallel mc cp invocations) ==" | tee -a "${RESULTS}"
# Real client-side concurrency sweep against the many-small-object
# webdataset shard set -- the workload shape step 2's small-file study
# showed is metadata-operation-heavy, so concurrency (parallel PUT
# requests) is the knob that matters here, not part size (these objects
# are single-part).
for parallel in 1 4 16; do
  echo "-- parallel=${parallel} --" | tee -a "${RESULTS}"
  time (
    find "${LOAD_DIR}/webdataset" -type f | \
      xargs -P "${parallel}" -I{} mc cp {} "${MINIO_ALIAS}/hpc-tuning-bucket/webdataset-p${parallel}/"
  ) 2>&1 | tee -a "${RESULTS}"
done

echo "== Knob 3 (config-time, not swept here): erasure-coding parity level ==" | tee -a "${RESULTS}"
echo "Set at deploy_minio.sh's server-startup time (drive count determines the default EC" | tee -a "${RESULTS}"
echo "parity level; explicit override via MINIO_STORAGE_CLASS_STANDARD=EC:2 / EC:4 env var" | tee -a "${RESULTS}"
echo "before 'minio server' starts) -- re-run deploy_minio.sh with each setting to compare." | tee -a "${RESULTS}"

echo "Results written to ${RESULTS}"
