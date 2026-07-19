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
  reorder)
    # netem reorder only takes effect on packets *delayed* relative to a
    # baseline (that's how netem models "some packets take a fast path"),
    # so this needs a base delay too — 10ms delay, 25% of packets skip it
    # (arrive immediately, i.e. out of order relative to the delayed 75%).
    PCT="${3:-25%}"
    tc qdisc add dev "$IFACE" root netem delay 10ms reorder "$PCT" 50%
    echo "Injected ${PCT} reordering on ${IFACE} (base delay 10ms)"
    ;;
  corrupt)
    RATE="${3:-1%}"
    tc qdisc add dev "$IFACE" root netem corrupt "$RATE"
    echo "Injected ${RATE} bit-corruption on ${IFACE}"
    ;;
  partition)
    # Full partition: 100% loss, indistinguishable from a severed link at
    # the application layer (as opposed to `loss 100%` phrased as a rate,
    # this is the same mechanism — kept as a distinct case purely for
    # inject.sh's own readability when scripted from chaos_run.sh).
    tc qdisc add dev "$IFACE" root netem loss 100%
    echo "Injected full partition (100% loss) on ${IFACE}"
    ;;
  restore)
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    echo "Restored ${IFACE}"
    ;;
  *)
    echo "Unknown fault: $FAULT. Options: latency, loss, reorder, corrupt, partition, restore"
    exit 1
    ;;
esac
