#!/usr/bin/env bash
# deploy_minio.sh — PLAN.md Phase 21 step 11. Real, complete MinIO
# deployment commands (erasure-coded 4-drive local server + `mc` alias/
# bucket setup), unrun -- MinIO is not installed locally (the user
# explicitly declined this session's install offer: "none of these right
# now, just write the code and note these need to be installed
# eventually" -- see hpc_storage/README.md). Same toolchain-gated-but-
# unrun convention as gpu_engine's CUDA code or fpga_engine's HLS/TCL.
#
# Install (when granted): `brew install minio/stable/minio
# minio/stable/mc` (single-binary, no Docker needed on macOS).
set -euo pipefail

DATA_ROOT="${1:-/tmp/hpc_storage_minio_data}"
MINIO_ALIAS="hpcstorage"

echo "== Step 1: start a 4-drive erasure-coded single-node MinIO server =="
# Real MinIO erasure-coding syntax: 4 local directories as one EC set
# (MinIO auto-selects EC:2 parity for a 4-drive set -- tolerates 1 drive
# loss with this drive count, 2 with EC:4 explicitly). A real multi-node
# deployment would instead pass drive paths across several hosts
# (http://node{1...4}/data), which needs the real cluster this Mac
# doesn't have -- this is the honest single-node local substitute step 11
# itself documents (see README.md's hardware-access note).
mkdir -p "${DATA_ROOT}"/{1,2,3,4}
minio server "${DATA_ROOT}"/{1,2,3,4} --console-address ":9090" &
MINIO_PID=$!
sleep 2

echo "== Step 2: configure mc alias + create the tuning bucket =="
mc alias set "${MINIO_ALIAS}" http://127.0.0.1:9000 minioadmin minioadmin
mc mb "${MINIO_ALIAS}/hpc-tuning-bucket"

echo "== Step 3: real tunable server config -- knobs step 11 actually turns =="
# api_requests_max: bounds concurrent in-flight S3 API requests (the
# server-side half of the concurrent-connection-limit knob
# tune_and_measure.sh sweeps from the client side).
mc admin config set "${MINIO_ALIAS}" api api_requests_max=1600 api_requests_deadline=10s
mc admin service restart "${MINIO_ALIAS}"

echo "MinIO server running as PID ${MINIO_PID}. Run tune_and_measure.sh next."
