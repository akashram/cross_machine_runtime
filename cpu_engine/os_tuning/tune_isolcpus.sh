#!/usr/bin/env bash
# =============================================================================
# tune_isolcpus.sh — isolate CPUs from the kernel scheduler
#
# WHAT IT DOES
# ------------
# Adds kernel boot parameters that tell Linux to never schedule ordinary tasks
# on the specified CPUs unless a process explicitly requests it via CPU affinity.
#
#   isolcpus=<cpus>
#     Removes the listed CPUs from the kernel scheduler's runqueue. Tasks are
#     never placed there by the scheduler. The only way to run on an isolated
#     CPU is via sched_setaffinity()/pthread_setaffinity_np() — exactly what
#     ThreadPinner::pin() does.
#
#   nohz_full=<cpus>
#     Disables the kernel's periodic "tick" on isolated CPUs when only one
#     task is running there. Without this, Linux fires a timer interrupt every
#     1/HZ seconds (default HZ=250 → every 4 ms) even on idle-looking CPUs.
#     Each tick adds ~2-5 us of interrupt latency and can delay a sleeping
#     thread by a full tick period. nohz_full eliminates this — the CPU fires
#     its timer interrupt only when the next real deadline arrives.
#
#   rcu_nocbs=<cpus>
#     RCU (Read-Copy-Update) callbacks are scheduled work items that Linux
#     fires on every CPU periodically to reclaim memory from lock-free
#     data structures. Each callback is a ~10-50 us interrupt. rcu_nocbs
#     offloads these callbacks to a dedicated "rcuob" kernel thread on the
#     housekeeping CPU, preventing them from running on isolated CPUs.
#
# REQUIRES REBOOT
# ---------------
# Kernel cmdline parameters take effect only after the next boot.
# All other tune_*.sh scripts can be applied without rebooting.
#
# REVERTING
# ---------
# Remove the parameters from GRUB_CMDLINE_LINUX_DEFAULT and run
# update-grub, then reboot. Or: restore_all.sh covers the runtime
# state (IRQ, C-states, governor) without needing a reboot.
#
# USAGE
# -----
#   sudo bash tune_isolcpus.sh [cpu_list]
#   cpu_list: comma-separated CPU indices or ranges, e.g. "2,3" or "2-7"
#   Default: all CPUs except CPU 0 (leave CPU 0 as housekeeping)
#
# EXAMPLE (4-core machine — isolate CPUs 1,2,3; keep CPU 0 for OS):
#   sudo bash tune_isolcpus.sh 1-3
#
# VERIFICATION (after reboot)
#   cat /sys/devices/system/cpu/isolated
#   taskset -c 1 sleep 10 &    # will stick to CPU 1
#   ps -eo pid,psr,comm | grep -v " 1 "  # no normal tasks on CPU 1
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if ! uname -r | grep -q linux 2>/dev/null && [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: this script is Linux-only"
    echo "macOS has no kernel scheduler isolation API."
    echo "On macOS, thread_policy_set() provides advisory affinity hints only."
    exit 1
fi

# Determine CPU list
TOTAL_CPUS=$(nproc --all)
if [[ $# -ge 1 ]]; then
    CPU_LIST="$1"
else
    # Default: isolate everything except CPU 0
    if [[ "$TOTAL_CPUS" -le 1 ]]; then
        echo "ERROR: only 1 CPU available; need at least 2 to isolate any"
        exit 1
    fi
    CPU_LIST="1-$((TOTAL_CPUS - 1))"
fi

echo "=== tune_isolcpus.sh ==="
echo "CPUs to isolate: $CPU_LIST (housekeeping: CPU 0)"
echo "Total CPUs: $TOTAL_CPUS"
echo ""
echo "NOTE: This script MODIFIES GRUB and requires a REBOOT to take effect."
echo "      All other tune_*.sh scripts take effect immediately without reboot."
echo ""

# Show current state
CURRENT=""
[[ -f /sys/devices/system/cpu/isolated ]] && CURRENT=$(cat /sys/devices/system/cpu/isolated)
echo "Current isolated CPUs: '${CURRENT:-none}'"
echo ""

GRUB_FILE="/etc/default/grub"
if [[ ! -f "$GRUB_FILE" ]]; then
    echo "ERROR: $GRUB_FILE not found. Not a GRUB-based system?"
    exit 1
fi

# Backup grub config
cp "$GRUB_FILE" "${GRUB_FILE}.bak.$(date +%s)"
echo "Backed up $GRUB_FILE"

PARAMS="isolcpus=$CPU_LIST nohz_full=$CPU_LIST rcu_nocbs=$CPU_LIST"

# Check if already present
if grep -q "isolcpus" "$GRUB_FILE"; then
    echo "WARNING: isolcpus already present in $GRUB_FILE"
    echo "Current line:"
    grep "GRUB_CMDLINE_LINUX" "$GRUB_FILE"
    echo ""
    echo "Remove existing isolcpus/nohz_full/rcu_nocbs manually, then re-run."
    exit 1
fi

# Add params to GRUB_CMDLINE_LINUX_DEFAULT
sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"${PARAMS} /" "$GRUB_FILE"

echo "Added to $GRUB_FILE:"
grep "GRUB_CMDLINE_LINUX" "$GRUB_FILE"
echo ""

# Regenerate GRUB config
if command -v update-grub &>/dev/null; then
    update-grub
elif command -v grub2-mkconfig &>/dev/null; then
    grub2-mkconfig -o /boot/grub2/grub.cfg
else
    echo "WARNING: could not find update-grub or grub2-mkconfig"
    echo "Regenerate GRUB config manually before rebooting"
fi

echo ""
echo "=== NEXT STEPS ==="
echo "1. Reboot: sudo reboot"
echo "2. Verify: cat /sys/devices/system/cpu/isolated"
echo "3. Then run: sudo bash tune_irq_affinity.sh $CPU_LIST"
echo "             sudo bash tune_cstate.sh $CPU_LIST"
echo "             sudo bash tune_governor.sh $CPU_LIST"
echo "   Or:       sudo bash tune_all.sh $CPU_LIST"
