#!/usr/bin/env bash
# =============================================================================
# tune_all.sh — apply all runtime OS tuning for latency-sensitive workloads
#
# WHAT THIS DOES
# --------------
# Applies all OS-level tuning that does NOT require a reboot, in the correct
# order. For isolcpus/nohz_full (which require a reboot), see tune_isolcpus.sh.
#
# Order matters:
#   1. Governor first — must be at max frequency before running measurements
#   2. C-states — disable deep sleep on isolated CPUs
#   3. IRQ affinity — move interrupts off isolated CPUs last (stops irqbalance)
#
# EXPECTED LATENCY IMPROVEMENT (p99 scheduling jitter, 200 us target sleep)
# -------------------------------------------------------------------------
#   Baseline (no tuning):               p99  500–5000 us
#   After tune_all.sh:                  p99   20–100  us
#   After tune_all.sh + isolcpus reboot:p99    2–10   us
#
# USAGE
# -----
#   sudo bash tune_all.sh [cpu_list]
#   cpu_list: isolated CPUs, e.g. "1-3" or "1,3" (default: all except CPU 0)
#
# MEASURE BEFORE/AFTER
#   ./jitter_bench <cpu>                # baseline
#   sudo bash tune_all.sh <cpu_list>
#   ./jitter_bench <cpu>                # after tuning
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Linux only. macOS has no kernel-level CPU isolation."
    echo ""
    echo "macOS provides:"
    echo "  - thread_policy_set(THREAD_AFFINITY_POLICY): advisory scheduling hint"
    echo "  - No C-state control (hardware-managed via DVFS)"
    echo "  - No IRQ affinity control (/proc/irq does not exist)"
    echo "  - No CPU governor (HW-P-states on Intel, custom DVFS on Apple Silicon)"
    echo ""
    echo "For latency testing on macOS, ThreadPinner::pin() is the only lever."
    echo "Real isolation numbers require a Linux instance (AWS c5.2xlarge etc.)."
    exit 1
fi

TOTAL_CPUS=$(nproc --all)
if [[ $# -ge 1 ]]; then
    ISOLATED_LIST="$1"
else
    ISOLATED_LIST="1-$((TOTAL_CPUS - 1))"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JITTER_BENCH="${SCRIPT_DIR}/../../build/release/cpu_engine/bench/jitter_bench"

# Use first isolated CPU for measurement
FIRST_ISOLATED=$(echo "$ISOLATED_LIST" | tr ',' '\n' | head -1 | cut -d'-' -f1)

echo "=========================================="
echo "OS Latency Tuning — cpu_engine step 3"
echo "Isolated CPUs: $ISOLATED_LIST"
echo "Measurement CPU: $FIRST_ISOLATED"
echo "=========================================="
echo ""

# Baseline measurement (if jitter_bench is available)
if [[ -x "$JITTER_BENCH" ]]; then
    echo "--- BASELINE (before tuning) ---"
    "$JITTER_BENCH" "$FIRST_ISOLATED"
    echo ""
else
    echo "NOTE: jitter_bench not found at $JITTER_BENCH"
    echo "      Build with: cmake --preset release && cmake --build --preset release"
    echo ""
fi

# Step 1: CPU governor
echo "--- Step 1: CPU governor → performance ---"
bash "${SCRIPT_DIR}/tune_governor.sh" "0-$((TOTAL_CPUS-1))"
echo ""

# Step 2: C-states
echo "--- Step 2: Disable deep C-states ---"
bash "${SCRIPT_DIR}/tune_cstate.sh" "$ISOLATED_LIST"
echo ""

# Step 3: IRQ affinity
echo "--- Step 3: IRQ affinity ---"
bash "${SCRIPT_DIR}/tune_irq_affinity.sh" "$ISOLATED_LIST"
echo ""

# Post-tuning measurement
if [[ -x "$JITTER_BENCH" ]]; then
    echo "--- AFTER TUNING ---"
    "$JITTER_BENCH" "$FIRST_ISOLATED"
    echo ""
fi

echo "=========================================="
echo "Tuning applied. For best results also run:"
echo "  sudo bash tune_isolcpus.sh $ISOLATED_LIST"
echo "  sudo reboot"
echo "Then re-run tune_all.sh (to redo IRQ/cstate after reboot)."
echo ""
echo "To restore defaults: sudo bash restore_all.sh $ISOLATED_LIST"
echo "=========================================="
