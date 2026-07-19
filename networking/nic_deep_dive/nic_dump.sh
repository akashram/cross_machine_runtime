#!/usr/bin/env bash
# nic_dump.sh — dump the NIC configuration state this step's design doc
# is about: descriptor ring sizes, RSS (receive-side scaling) hashing
# config, PFC/ECN (lossless fabric) status, and hardware timestamping
# capability. Read-only — this is a diagnostic snapshot, not a tuning
# script (tuning changes belong in a separate, reviewed script once
# there's a real workload's numbers to tune against).
#
# Usage: ./nic_dump.sh <iface>
# TODO: run on Linux (ENA NIC on AWS, or any Linux NIC).

set -euo pipefail
IFACE="${1:?Usage: $0 <iface>}"

echo "== Descriptor ring sizes (ethtool -g) =="
ethtool -g "$IFACE" || echo "ethtool -g not supported by this driver"

echo
echo "== RSS: queue count + hash indirection table (ethtool -l / -x) =="
ethtool -l "$IFACE" || echo "ethtool -l not supported"
ethtool -x "$IFACE" || echo "ethtool -x not supported"

echo
echo "== RSS: hash key + fields (ethtool -X current, -n for flow rules) =="
ethtool -n "$IFACE" rx-flow-hash || echo "rx-flow-hash query not supported"

echo
echo "== PFC (Priority Flow Control) — per-priority pause state (ethtool -a / DCB) =="
ethtool -a "$IFACE" || echo "ethtool -a not supported"
if command -v dcb >/dev/null 2>&1; then
  dcb pfc show dev "$IFACE" || echo "dcb pfc show failed (DCBX not configured)"
else
  echo "dcb (iproute2) not installed — cannot query DCBX/PFC state"
fi

echo
echo "== ECN — RED/ECN qdisc state (tc -s qdisc) =="
tc -s qdisc show dev "$IFACE"

echo
echo "== Hardware timestamping capability (ethtool -T) =="
ethtool -T "$IFACE" || echo "ethtool -T not supported"

echo
echo "== IRQ affinity (per-queue IRQ -> CPU mapping) =="
for irq_dir in /sys/class/net/"$IFACE"/device/msi_irqs/*; do
  [ -d "$irq_dir" ] || continue
  irq=$(basename "$irq_dir")
  affinity=$(cat "/proc/irq/${irq}/smp_affinity_list" 2>/dev/null || echo "?")
  echo "irq ${irq}: cpu ${affinity}"
done
