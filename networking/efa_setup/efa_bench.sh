#!/usr/bin/env bash
# efa_bench.sh — install/validate EFA and run the fi_pingpong baseline
# between two EFA-enabled nodes. The TCP comparison column in this step's
# README isn't measured here with netcat — it's `rdma_v1/tcp_baseline_bench`,
# a real compiled benchmark over networking/common/TcpChannel, run with the
# same message sizes for a genuine apples-to-apples number (see
# rdma_v1/README.md). This script is EFA-only.
#
# Usage:
#   node1 (server): ./efa_bench.sh server
#   node2 (client): ./efa_bench.sh client <server_ip>
#
# TODO: run on EFA hardware (2x p4d.24xlarge, same VPC placement group).

set -euo pipefail

MODE="${1:?Usage: $0 <server|client|verify> [server_ip]}"
MSG_SIZES=(64 4096 1048576) # 64B, 4KB, 1MB — matches rdma_v1/README.md's table

verify_efa() {
  echo "== EFA device enumeration (fi_info) =="
  fi_info -p efa || { echo "no EFA provider found — is aws-efa-installer installed?"; exit 1; }
  echo
  echo "== EFA device status =="
  # ENA/EFA devices show up as extra NICs beyond the primary ENI.
  ibv_devinfo 2>/dev/null || echo "ibv_devinfo not found (rdma-core not installed?)"
}

run_pingpong_efa() {
  local role="$1" server_ip="${2:-}"
  for size in "${MSG_SIZES[@]}"; do
    echo "== fi_pingpong EFA, message size ${size}B =="
    if [[ "$role" == "server" ]]; then
      fi_pingpong -p efa -e rdm -S "$size"
    else
      fi_pingpong -p efa -e rdm -S "$size" "$server_ip"
    fi
  done
}

case "$MODE" in
  verify) verify_efa ;;
  server) run_pingpong_efa server ;;
  client) run_pingpong_efa client "${2:?Usage: $0 client <server_ip>}" ;;
  *) echo "Unknown mode: $MODE. Options: verify, server, client"; exit 1 ;;
esac
