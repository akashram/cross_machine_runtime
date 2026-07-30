#!/bin/bash
# run_local_demo.sh — launches 4 REAL, separate OS processes of
# training_worker on 127.0.0.1, one per rank, and waits for all of them
# to finish. This is the actual validation that training_worker's
# process-per-rank design works: real inter-process TCP (via
# networking/common's new peer_hosts TcpChannel constructor), not threads
# sharing one process the way every other multi-rank step in this repo
# (e.g. full_training_loop/training_loop_test.cpp) simulates ranks.
#
# Usage: run from the repo root after building the `training_worker`
# CMake target: ./distributed_training/training_worker/run_local_demo.sh
set -euo pipefail

BIN="${TRAINING_WORKER_BIN:-./build/debug/distributed_training/training_worker/training_worker}"
WORLD_SIZE=4
BASE_PORT=46301
CKPT_DIR="$(mktemp -d)"
LOG_DIR="$(mktemp -d)"
PEER_HOSTS="127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1"

echo "training_worker real multi-process demo: world_size=$WORLD_SIZE ckpt_dir=$CKPT_DIR"

pids=()
for r in $(seq 0 $((WORLD_SIZE - 1))); do
  RANK="$r" WORLD_SIZE="$WORLD_SIZE" PEER_HOSTS="$PEER_HOSTS" BASE_PORT="$BASE_PORT" \
    EPOCHS=30 CHECKPOINT_DIR="$CKPT_DIR" \
    "$BIN" > "$LOG_DIR/rank$r.log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "rank $i FAILED (see $LOG_DIR/rank$i.log)"
    status=1
  fi
done

for r in $(seq 0 $((WORLD_SIZE - 1))); do
  echo "--- rank $r log ---"
  cat "$LOG_DIR/rank$r.log"
done

rm -rf "$CKPT_DIR" "$LOG_DIR"
exit $status
