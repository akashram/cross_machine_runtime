#!/usr/bin/env bash
# =============================================================================
# restore_all.sh — restore OS defaults after tune_all.sh
#
# Reverses:
#   - CPU governor → ondemand (or schedutil on newer kernels)
#   - C-states → all re-enabled
#   - IRQ affinity → all CPUs (restart irqbalance)
#   - Turbo Boost → re-enabled
#
# Does NOT reverse:
#   - isolcpus/nohz_full/rcu_nocbs — these require editing /etc/default/grub
#     and rebooting. Run tune_isolcpus.sh to see instructions, then
#     remove the parameters from GRUB manually and reboot.
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Linux only."
    exit 1
fi

TOTAL_CPUS=$(nproc --all)

echo "=== restore_all.sh ==="
echo ""

# 1. Restore CPU governor to ondemand/schedutil
DEFAULT_GOVERNOR="ondemand"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors ]]; then
    AVAIL=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors)
    if [[ "$AVAIL" == *"schedutil"* ]]; then
        DEFAULT_GOVERNOR="schedutil"
    elif [[ "$AVAIL" == *"ondemand"* ]]; then
        DEFAULT_GOVERNOR="ondemand"
    else
        DEFAULT_GOVERNOR="powersave"
    fi
fi

echo "Restoring CPU governor → $DEFAULT_GOVERNOR"
for (( cpu=0; cpu<TOTAL_CPUS; cpu++ )); do
    gov_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
    [[ -f "$gov_file" ]] && echo "$DEFAULT_GOVERNOR" > "$gov_file" 2>/dev/null || true
done
echo "  Done."

# 2. Re-enable Turbo Boost
TURBO_FILE="/sys/devices/system/cpu/intel_pstate/no_turbo"
if [[ -f "$TURBO_FILE" ]]; then
    echo 0 > "$TURBO_FILE"
    echo "Re-enabled Intel Turbo Boost."
fi
AMD_BOOST="/sys/devices/system/cpu/cpufreq/boost"
if [[ -f "$AMD_BOOST" ]]; then
    echo 1 > "$AMD_BOOST" 2>/dev/null || true
    echo "Re-enabled AMD Precision Boost."
fi

# 3. Re-enable all C-states
echo "Re-enabling all C-states..."
REENABLED=0
for (( cpu=0; cpu<TOTAL_CPUS; cpu++ )); do
    cpuidle_dir="/sys/devices/system/cpu/cpu${cpu}/cpuidle"
    [[ -d "$cpuidle_dir" ]] || continue
    for state_dir in "${cpuidle_dir}"/state*/; do
        dis="${state_dir}disable"
        [[ -f "$dis" ]] && echo 0 > "$dis" 2>/dev/null && (( REENABLED++ )) || true
    done
done
echo "  Re-enabled $REENABLED C-state entries."

# 4. Restore IRQ affinity — let all CPUs handle all IRQs
echo "Restoring IRQ affinity (all CPUs)..."
ALL_MASK=$(printf "%x" $(( (1 << TOTAL_CPUS) - 1 )))
echo "$ALL_MASK" > /proc/irq/default_smp_affinity 2>/dev/null || true
MOVED=0
for aff_file in /proc/irq/[0-9]*/smp_affinity; do
    echo "$ALL_MASK" > "$aff_file" 2>/dev/null && (( MOVED++ )) || true
done
echo "  Restored $MOVED IRQ affinity entries."

# 5. Restart irqbalance
if command -v irqbalance &>/dev/null; then
    systemctl start irqbalance 2>/dev/null && echo "  irqbalance restarted." || \
        echo "  irqbalance start failed (may not be installed as a service)."
fi

echo ""
echo "DONE. All runtime tuning restored to defaults."
echo ""
echo "NOTE: isolcpus/nohz_full/rcu_nocbs kernel parameters are NOT removed here."
echo "      To remove them: edit /etc/default/grub, run update-grub, reboot."
