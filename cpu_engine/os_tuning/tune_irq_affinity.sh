#!/usr/bin/env bash
# =============================================================================
# tune_irq_affinity.sh — move all IRQs off isolated CPUs
#
# WHAT IT DOES
# ------------
# Every hardware interrupt (NIC, disk, timer, USB, ...) fires on some CPU.
# The Linux irqbalance daemon and the kernel's default spread interrupts
# across all available CPUs for throughput. For latency-sensitive workloads
# this is harmful — an IRQ landing on the measurement CPU adds 5–500 us of
# handler latency regardless of what the CPU was doing.
#
# /proc/irq/<N>/smp_affinity is a CPU bitmask (hex) that controls which CPUs
# may handle IRQ N. Setting it to only housekeeping CPUs means isolated CPUs
# never get interrupted.
#
# /proc/irq/default_smp_affinity controls the default mask for new IRQs.
#
# WHAT REMAINS ON ISOLATED CPUS
# ------------------------------
# Even after this script, isolated CPUs still receive:
#   - Local APIC timer interrupts (needed for kernel timekeeping; unavoidable
#     without nohz_full, which suppresses most of them)
#   - IPIs (inter-processor interrupts) from other CPUs — e.g. TLB shootdowns
#     These are rare and low-latency (~1 us) in normal operation
#
# KILLS irqbalance
# ----------------
# irqbalance is a daemon that continuously reassigns IRQ affinity for
# throughput. It will immediately undo this script's changes unless stopped.
# This script stops it. Restore it with restore_all.sh.
#
# USAGE
# -----
#   sudo bash tune_irq_affinity.sh [cpu_list]
#   cpu_list: comma-separated isolated CPU indices, e.g. "1,2,3" or "1-3"
#   Default: all CPUs except CPU 0
#
# VERIFICATION
#   cat /proc/irq/*/smp_affinity_list | sort -u
#   # should only show CPU 0 (or your housekeeping set)
# =============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Linux only. macOS has no /proc/irq interface."
    exit 1
fi

TOTAL_CPUS=$(nproc --all)
if [[ $# -ge 1 ]]; then
    ISOLATED_LIST="$1"
else
    ISOLATED_LIST="1-$((TOTAL_CPUS - 1))"
fi

echo "=== tune_irq_affinity.sh ==="
echo "Isolated CPUs: $ISOLATED_LIST"

# Build the housekeeping CPU mask (all CPUs NOT in the isolated list).
# We compute the bitmask as a hex value for /proc/irq/*/smp_affinity.
# Strategy: start with all-ones mask, then clear isolated CPU bits.
HOUSEKEEPING_MASK=0
for (( cpu=0; cpu<TOTAL_CPUS; cpu++ )); do
    HOUSEKEEPING_MASK=$(( HOUSEKEEPING_MASK | (1 << cpu) ))
done

# Parse isolated CPU list into an array and clear those bits.
# Handles ranges like "1-3" and comma-separated like "1,3,5".
parse_cpulist() {
    local list="$1"
    local -a cpus=()
    IFS=',' read -ra parts <<< "$list"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for (( c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++ )); do
                cpus+=("$c")
            done
        else
            cpus+=("$part")
        fi
    done
    echo "${cpus[@]}"
}

ISOLATED_CPUS=( $(parse_cpulist "$ISOLATED_LIST") )
for cpu in "${ISOLATED_CPUS[@]}"; do
    HOUSEKEEPING_MASK=$(( HOUSEKEEPING_MASK & ~(1 << cpu) ))
done

HK_HEX=$(printf "%x" "$HOUSEKEEPING_MASK")
echo "Housekeeping mask: 0x${HK_HEX} (CPUs keeping IRQs)"
echo ""

# Stop irqbalance — it will undo affinity changes if left running
if systemctl is-active --quiet irqbalance 2>/dev/null; then
    echo "Stopping irqbalance daemon..."
    systemctl stop irqbalance
    echo "  irqbalance stopped. restore_all.sh will restart it."
fi

# Set default affinity for new IRQs
echo "$HK_HEX" > /proc/irq/default_smp_affinity
echo "Set /proc/irq/default_smp_affinity = 0x${HK_HEX}"

# Apply to all existing IRQs
MOVED=0
SKIPPED=0
for irq_dir in /proc/irq/[0-9]*/; do
    irq=$(basename "$irq_dir")
    aff_file="${irq_dir}smp_affinity"

    [[ -f "$aff_file" ]] || continue

    # Some IRQs (e.g. local APIC) cannot be changed and will error; skip them.
    if ! echo "$HK_HEX" > "$aff_file" 2>/dev/null; then
        (( SKIPPED++ )) || true
        continue
    fi
    (( MOVED++ )) || true
done

echo "Moved $MOVED IRQs to housekeeping CPUs. Skipped $SKIPPED (local APIC etc.)"
echo ""

# Verification
echo "Sample IRQ affinity after change:"
for irq_dir in /proc/irq/[0-9]*/; do
    irq=$(basename "$irq_dir")
    aff_list="${irq_dir}smp_affinity_list"
    [[ -f "$aff_list" ]] || continue
    desc_file="${irq_dir}../name"
    name=""
    [[ -f "${irq_dir}../../../irq_name" ]] && name=$(cat "${irq_dir}../../../irq_name" 2>/dev/null) || true
    printf "  IRQ %3s: CPUs %s\n" "$irq" "$(cat "$aff_list")"
done | head -20

echo ""
echo "DONE. Verify: cat /proc/irq/*/smp_affinity_list | sort -u"
echo "To restore:  sudo bash restore_all.sh"
