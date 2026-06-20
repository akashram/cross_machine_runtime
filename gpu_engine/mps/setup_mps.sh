#!/usr/bin/env bash
# setup_mps.sh — enable CUDA MPS server for multi-process GPU sharing
# Run as root (or with sudo)
# TODO: run on Linux GPU hardware

set -euo pipefail

GPU_ID="${1:-0}"

echo "[MPS] Setting GPU $GPU_ID to EXCLUSIVE_PROCESS mode..."
nvidia-smi -i "$GPU_ID" -c EXCLUSIVE_PROCESS

echo "[MPS] Starting MPS daemon..."
nvidia-cuda-mps-control -d

echo "[MPS] MPS is running. Verify with: nvidia-smi | grep MPS"
echo "[MPS] To stop: echo quit | nvidia-cuda-mps-control"
echo "[MPS] To reset mode: nvidia-smi -i $GPU_ID -c DEFAULT"
