#!/usr/bin/env bash
# inject.sh — inject network faults via tc netem
# Usage: ./inject.sh <iface> <fault_type> [params...]
# TODO: run on Linux with root privileges

set -euo pipefail
IFACE="${1:?Usage: $0 <iface> <fault>}"
FAULT="${2:?}"

case "$FAULT" in
  latency)
    DELAY="${3:-100ms}"
    tc qdisc add dev "$IFACE" root netem delay "$DELAY" "$DELAY"
    echo "Injected ${DELAY} latency on ${IFACE}"
    ;;
  loss)
    RATE="${3:-10%}"
    tc qdisc add dev "$IFACE" root netem loss "$RATE"
    echo "Injected ${RATE} packet loss on ${IFACE}"
    ;;
  restore)
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    echo "Restored ${IFACE}"
    ;;
  *)
    echo "Unknown fault: $FAULT. Options: latency, loss, restore"
    exit 1
    ;;
esac
