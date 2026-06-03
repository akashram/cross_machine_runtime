#!/usr/bin/env bash
# =============================================================================
# tune_governor.sh — set CPU frequency governor to "performance"
#
# WHAT IS A CPU GOVERNOR
# ----------------------
# Modern CPUs can run at variable frequencies (P-states) to trade power for
# performance. The Linux cpufreq subsystem uses a "governor" to decide which
# frequency to use at any moment:
#
#   powersave    — always use minimum frequency. Maximises battery life.
#   ondemand     — default on servers. Ramps up frequency when load is detected,
#                  drops back after an idle period. Ramp-up takes ~10-50 ms.
#   conservative — like ondemand but ramps more slowly.
#   schedutil    — frequency driven by scheduler utilisation signals (newer kernels).
#   performance  — always use maximum frequency. No frequency changes ever.
#
# WHY "ONDEMAND" HURTS LATENCY
# ----------------------------
# A thread waking from sleep may find the CPU at minimum frequency if it was
# recently idle. The governor takes ~10-50 ms to detect high load and ramp up.
# During that ramp-up window, each instruction takes 2-5x longer than expected.
# This shows up as a large spike in the first few iterations after a sleep.
#
# The "performance" governor fixes this by locking the CPU at maximum frequency.
# For power-sensitive deployments, consider "ondemand" on housekeeping CPUs and
# "performance" only on the isolated latency-critical CPUs (this script supports
# per-CPU governor setting).
#
# TURBO BOOST
# -----------
# Intel Turbo Boost (AMD Precision Boost) allows CPUs to temporarily exceed
# their rated "base" frequency when thermal headroom is available. This is
# desirable for throughput but can cause latency variance — the CPU may run at
# different frequencies on different iterations depending on thermal state.
#
# For the most reproducible benchmark numbers, disable turbo:
#   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# This trades ~10-30% throughput for ~5% lower jitter variance. restore_all.sh
# re-enables turbo.
#
# HARDWARE REQUIREMENTS
# ---------------------
# Requires the cpufreq driver to be loaded. On systems with the intel_pstate
# driver (Broadwell+), the governors available are limited to "performance" and
# "powersave" (the driver implements P-state control internally).
# On acpi-cpufreq systems, all governors above are available.
#
# Check:  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
#
# USAGE
# -----
#   sudo bash tune_governor.sh [cpu_list]
#   cpu_list: comma-separated CPU indices, e.g. "1,2,3" or "1-3"
#   Default: all CPUs
#
# VERIFICATION
#   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq  # current MHz * 1000
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Linux only."
    echo "macOS: CPU frequency is managed by the hardware PMU (HW-P-states on Intel"
    echo "       or DVFS on Apple Silicon). No user-space governor control."
    exit 1
fi

TOTAL_CPUS=$(nproc --all)
if [[ $# -ge 1 ]]; then
    CPU_LIST="$1"
else
    CPU_LIST="0-$((TOTAL_CPUS - 1))"
fi

TARGET_GOVERNOR="${2:-performance}"

echo "=== tune_governor.sh ==="
echo "CPUs: $CPU_LIST  →  governor: $TARGET_GOVERNOR"
echo ""

# Parse CPU list
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

# Show current state
echo "Current governors:"
for cpu in $(parse_cpulist "$CPU_LIST"); do
    gov_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
    [[ -f "$gov_file" ]] && printf "  CPU %-3s: %s\n" "$cpu" "$(cat "$gov_file")"
done | head -8
echo ""

# Check if target governor is available
FIRST_CPU=$(parse_cpulist "$CPU_LIST" | head -1)
AVAIL_FILE="/sys/devices/system/cpu/cpu${FIRST_CPU}/cpufreq/scaling_available_governors"
if [[ -f "$AVAIL_FILE" ]]; then
    AVAILABLE=$(cat "$AVAIL_FILE")
    if [[ " $AVAILABLE " != *" $TARGET_GOVERNOR "* ]]; then
        echo "ERROR: governor '$TARGET_GOVERNOR' not available on this system."
        echo "Available: $AVAILABLE"
        echo "On intel_pstate systems, only 'performance' and 'powersave' are valid."
        exit 1
    fi
fi

# Apply governor to each CPU
APPLIED=0
SKIPPED=0
for cpu in $(parse_cpulist "$CPU_LIST"); do
    gov_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
    if [[ -f "$gov_file" ]]; then
        echo "$TARGET_GOVERNOR" > "$gov_file"
        (( APPLIED++ )) || true
    else
        (( SKIPPED++ )) || true
    fi
done

echo "Set governor='$TARGET_GOVERNOR' on $APPLIED CPUs. ($SKIPPED CPUs had no cpufreq)"
echo ""

# Optionally disable Turbo Boost for maximum reproducibility
TURBO_FILE="/sys/devices/system/cpu/intel_pstate/no_turbo"
if [[ -f "$TURBO_FILE" ]]; then
    CURRENT_TURBO=$(cat "$TURBO_FILE")
    echo "Intel Turbo Boost: no_turbo=${CURRENT_TURBO}"
    echo "  (set to 1 to disable turbo for more reproducible benchmark numbers)"
    echo "  Command: echo 1 | sudo tee $TURBO_FILE"
else
    AMD_BOOST="/sys/devices/system/cpu/cpufreq/boost"
    if [[ -f "$AMD_BOOST" ]]; then
        echo "AMD Precision Boost: boost=$(cat "$AMD_BOOST")"
        echo "  (set to 0 to disable for reproducibility)"
    fi
fi
echo ""

# Show frequencies after change
echo "Frequencies after setting governor (MHz):"
for cpu in $(parse_cpulist "$CPU_LIST"); do
    freq_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq"
    max_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq"
    if [[ -f "$freq_file" ]]; then
        cur_mhz=$(( $(cat "$freq_file") / 1000 ))
        max_mhz=$(( $(cat "$max_file") / 1000 ))
        printf "  CPU %-3s: %4d MHz  (max: %d MHz)\n" "$cpu" "$cur_mhz" "$max_mhz"
    fi
done | head -8
echo ""

echo "DONE."
echo "To restore: sudo bash restore_all.sh"
