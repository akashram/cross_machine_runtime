#!/usr/bin/env bash
# node_kill.sh — kill a running process by PID (or the first match of a
# name pattern) with SIGKILL, simulating a hard node crash (as opposed to
# a graceful SIGTERM shutdown, which every component in this project
# already handles cleanly — the interesting failure mode for a chaos
# harness is the *ungraceful* one: no chance to flush state, close
# sockets, or notify peers).
#
# Usage: ./node_kill.sh <pid|name-pattern>
# TODO: run on a live multi-node cluster (e.g. raft_test-style processes,
# one per rank, actually running as separate OS processes rather than
# in-process threads — see networking/raft/README.md's note that this
# project's local tests simulate multi-node with threads, not processes).

set -euo pipefail
TARGET="${1:?Usage: $0 <pid|name-pattern>}"

if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  PID="$TARGET"
else
  PID=$(pgrep -f "$TARGET" | head -1) || { echo "no process matching '$TARGET'"; exit 1; }
fi

echo "killing PID ${PID} (SIGKILL, no graceful shutdown)"
kill -9 "$PID"
echo "killed at $(date +%s.%N)"
