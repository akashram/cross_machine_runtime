#!/usr/bin/env bash
# chaos_run.sh — orchestrates one fault-injection cycle end to end:
# inject a fault, wait for the cluster to notice, measure how long until
# it's healthy again, restore, and report. This is the harness the
# README's Results table comes from; individual fault mechanics live in
# inject.sh (network), node_kill.sh (process), gpu_oom.sh (GPU memory).
#
# Usage: ./chaos_run.sh <fault> <iface> [health_check_cmd]
#   fault: latency | loss | reorder | corrupt | partition | node_kill
#   health_check_cmd: a command that exits 0 when the cluster is healthy
#     again (e.g. "raft_client leader-of <cluster>" or a curl against a
#     health endpoint) — polled every 200ms after injection.
#
# TODO: run on a live multi-node cluster. This project's local tests
# (raft_test, chandy_lamport_test) simulate multi-node with threads in
# one process specifically so they're runnable without that cluster —
# this script is the next step once one exists.

set -euo pipefail
FAULT="${1:?Usage: $0 <fault> <iface> [health_check_cmd]}"
IFACE="${2:?}"
HEALTH_CHECK="${3:-true}" # default: no-op check, always "healthy" — replace with a real probe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== baseline: confirming cluster healthy before injection =="
if ! eval "$HEALTH_CHECK"; then
  echo "cluster not healthy before injection — aborting (a fault-recovery"
  echo "measurement is meaningless if it wasn't healthy to begin with)"
  exit 1
fi

INJECT_TIME=$(date +%s.%N)
case "$FAULT" in
  node_kill)
    "${SCRIPT_DIR}/node_kill.sh" "${4:?node_kill needs a pid/name pattern as \$4}"
    ;;
  *)
    "${SCRIPT_DIR}/inject.sh" "$IFACE" "$FAULT"
    ;;
esac
echo "fault '${FAULT}' injected at ${INJECT_TIME}"

echo "== polling health check until recovery (200ms interval) =="
RECOVERED_TIME=""
for _ in $(seq 1 150); do # 30s max wait
  if eval "$HEALTH_CHECK"; then
    RECOVERED_TIME=$(date +%s.%N)
    break
  fi
  sleep 0.2
done

if [[ "$FAULT" != "node_kill" ]]; then
  "${SCRIPT_DIR}/inject.sh" "$IFACE" restore
fi

if [[ -z "$RECOVERED_TIME" ]]; then
  echo "RESULT: fault=${FAULT} recovery=TIMEOUT (30s) — did not recover"
  exit 1
fi

RECOVERY_SECONDS=$(echo "$RECOVERED_TIME - $INJECT_TIME" | bc)
echo "RESULT: fault=${FAULT} recovery=${RECOVERY_SECONDS}s"
