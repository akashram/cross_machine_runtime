#!/usr/bin/env bash
# gpu_oom.sh — allocate GPU memory until cudaMalloc fails, to exercise
# whatever OOM-handling this project's GPU-side components have (Phase 3
# gpu_engine/'s memory management, step 2; the router's device-cost
# model, compiler/cost_model, treats a device as unusable once it can't
# satisfy an allocation). Requires nvidia-smi + a CUDA toolchain — same
# hardware gate as the rest of gpu_engine/.
#
# Usage: ./gpu_oom.sh [device_id]
# TODO: run on a GPU instance (gpu_engine/'s target hardware).

set -euo pipefail
DEVICE="${1:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — no GPU on this machine, nothing to inject"
  exit 1
fi

echo "== GPU ${DEVICE} memory before injection =="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -i "$DEVICE"

# A small CUDA program is the actual injector — allocating from a shell
# script alone isn't possible. This expects gpu_engine/'s memory step
# (Phase 3 step 2, gpu_alloc) to provide a `gpu_oom_inject` test binary;
# until that's built on real GPU hardware, this documents the intended
# invocation rather than duplicating a CUDA allocator loop here.
BINARY="$(dirname "$0")/../../gpu_engine/memory/gpu_oom_inject"
if [[ -x "$BINARY" ]]; then
  "$BINARY" --device "$DEVICE"
else
  echo "gpu_oom_inject not built — this requires gpu_engine/ built on GPU hardware first"
  echo "(would run: cudaMalloc in a loop until failure, report peak allocated bytes and error code)"
  exit 1
fi
