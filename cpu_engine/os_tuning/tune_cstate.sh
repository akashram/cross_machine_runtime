#!/usr/bin/env bash
# =============================================================================
# tune_cstate.sh — disable deep CPU C-states on isolated CPUs
#
# WHAT ARE C-STATES
# -----------------
# When a CPU core has no work to do, it enters a "C-state" power-saving mode:
#
#   C0  — fully active, executing instructions
#   C1  — "halt": CPU clock gated, very fast exit (~1 us)
#   C1E — enhanced halt: additional power reduction, ~2 us exit
#   C2  — "stop clock": larger power reduction, ~10-50 us exit latency
#   C3  — "sleep": cache flushed to L2, ~50-150 us exit latency
#   C6  — "deep power down": core voltage reduced, ~100-300 us exit latency
#   C7+ — package-level sleep: shared L3 flushed, 200-500 us exit latency
#
# Exact latencies depend on the microarchitecture (Skylake, Ice Lake, etc.)
# and BIOS settings. Check /sys/devices/system/cpu/cpu0/cpuidle/state*/latency
# for the kernel's estimate of each state's exit latency on your machine.
#
# WHY IT MATTERS
# --------------
# A thread sleeping for 200 us may find the CPU in C6 when it wakes up.
# Before executing even one instruction, the CPU must exit C6 (~200 us).
# Total wakeup latency: target + C-state exit = 200 + 200 = 400 us. This
# doubles the effective scheduling latency and creates large, irregular spikes
# in jitter_bench p99/p999 numbers.
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Disables all C-states deeper than C1 on the isolated CPUs by writing "1"
# to /sys/devices/system/cpu/cpu<N>/cpuidle/state<M>/disable.
# The CPU can still enter C1/C1E (very low exit latency), so power consumption
# only increases slightly vs. leaving deep C-states enabled.
#
# This is preferable to disabling ALL C-states (including C1) via the "idle=poll"
# kernel parameter, which forces the CPU to spin at 100% even when idle,
# consuming full power and generating heat that can trigger thermal throttling.
#
# ALTERNATIVE: intel_idle.max_cstate=1 kernel parameter
# This achieves the same effect system-wide without per-CPU granularity.
# Use this script for per-CPU control (isolate some cores, leave others free).
#
# USAGE
# -----
#   sudo bash tune_cstate.sh [cpu_list]
#   cpu_list: comma-separated isolated CPU indices, e.g. "1,2,3" or "1-3"
#   Default: all CPUs except CPU 0
#
# VERIFICATION
#   cat /sys/devices/system/cpu/cpu1/cpuidle/state*/disable
#   # 0 = enabled, 1 = disabled; C0/C1 should be 0, C2+ should be 1
#
#   cat /sys/devices/system/cpu/cpu1/cpuidle/state*/name
#   # shows the name of each state (POLL, C1, C1E, C2, C3, C6, ...)
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Linux only."
    echo "macOS: C-states are managed entirely by XNU; no user-space control."
    echo "Apple Silicon (M-series) implements DVFS differently from x86 C-states."
    exit 1
fi

TOTAL_CPUS=$(nproc --all)
if [[ $# -ge 1 ]]; then
    ISOLATED_LIST="$1"
else
    ISOLATED_LIST="1-$((TOTAL_CPUS - 1))"
fi

echo "=== tune_cstate.sh ==="
echo "Isolated CPUs: $ISOLATED_LIST"
echo ""

# Parse CPU list (supports "1-3" and "1,3,5")
parse_cpulist() {
    local list="$1"
    IFS=',' read -ra parts <<< "$list"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for (( c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++ )); do
                echo "$c"
            done
        else
            echo "$part"
        fi
    done
}

# Show current state table for first isolated CPU
FIRST_CPU=$(parse_cpulist "$ISOLATED_LIST" | head -1)
if [[ -d "/sys/devices/system/cpu/cpu${FIRST_CPU}/cpuidle" ]]; then
    echo "C-state table for CPU ${FIRST_CPU} (before):"
    printf "  %-6s %-12s %-10s %s\n" "State" "Name" "Latency(us)" "Disabled"
    for state_dir in /sys/devices/system/cpu/cpu${FIRST_CPU}/cpuidle/state*/; do
        state=$(basename "$state_dir")
        name=$(cat "${state_dir}name"    2>/dev/null || echo "?")
        lat=$(cat  "${state_dir}latency" 2>/dev/null || echo "?")
        dis=$(cat  "${state_dir}disable" 2>/dev/null || echo "?")
        printf "  %-6s %-12s %-10s %s\n" "$state" "$name" "$lat" "$dis"
    done
    echo ""
else
    echo "NOTE: /sys/devices/system/cpu/cpu${FIRST_CPU}/cpuidle not found"
    echo "      cpuidle may be disabled (kernel param: idle=poll or cpuidle.off=1)"
fi

# Apply: disable all states with exit latency > C1 (latency > ~2 us)
DISABLED=0
while IFS= read -r cpu; do
    cpuidle_dir="/sys/devices/system/cpu/cpu${cpu}/cpuidle"
    [[ -d "$cpuidle_dir" ]] || continue

    for state_dir in "${cpuidle_dir}"/state*/; do
        [[ -f "${state_dir}disable" ]] || continue
        [[ -f "${state_dir}latency" ]] || continue

        latency=$(cat "${state_dir}latency" 2>/dev/null || echo 0)
        name=$(cat "${state_dir}name" 2>/dev/null || echo "")

        # Keep C0 (POLL), C1, C1E — disable everything with latency > 2 us.
        # "POLL" state is the spin-idle state; its "latency" is 0 — always keep.
        if [[ "$latency" -gt 2 ]]; then
            if ! echo 1 > "${state_dir}disable" 2>/dev/null; then
                echo "  WARNING: could not disable ${state_dir} (${name})"
                continue
            fi
            (( DISABLED++ )) || true
        fi
    done
done < <(parse_cpulist "$ISOLATED_LIST")

echo "Disabled $DISABLED deep C-state entries across isolated CPUs."
echo ""

# Verify
echo "C-state table for CPU ${FIRST_CPU} (after):"
if [[ -d "/sys/devices/system/cpu/cpu${FIRST_CPU}/cpuidle" ]]; then
    printf "  %-6s %-12s %-10s %s\n" "State" "Name" "Latency(us)" "Disabled"
    for state_dir in /sys/devices/system/cpu/cpu${FIRST_CPU}/cpuidle/state*/; do
        state=$(basename "$state_dir")
        name=$(cat "${state_dir}name"    2>/dev/null || echo "?")
        lat=$(cat  "${state_dir}latency" 2>/dev/null || echo "?")
        dis=$(cat  "${state_dir}disable" 2>/dev/null || echo "?")
        printf "  %-6s %-12s %-10s %s\n" "$state" "$name" "$lat" "$dis"
    done
fi

echo ""
echo "DONE. C-states with latency > 2 us disabled on CPUs: $ISOLATED_LIST"
echo "To restore: sudo bash restore_all.sh"
