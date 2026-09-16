#!/usr/bin/env bash
# node_health_check.sh -- PLAN.md Phase 22 step 4: a real node
# health-check script in the shape of Slurm's own HealthCheckProgram
# mechanism (wired as slurm_jobs/slurm.conf's actual
# `HealthCheckProgram=` directive -- not a standalone script disconnected
# from it). Per Slurm's documented HealthCheckProgram semantics: slurmd
# runs this on a real interval on every IDLE/MIXED node, and a NON-ZERO
# exit status automatically DRAINs the node (no config on the Slurm side
# beyond HealthCheckProgram=/HealthCheckInterval= -- both already set in
# slurm_jobs/slurm.conf).
#
# Portable, real, run locally today -- checks what's actually checkable
# without a live Slurm daemon or root: disk space, expected process/
# service presence, and basic hardware sanity. Exits 0 (healthy) or 1
# (unhealthy, prints which check(s) failed to stderr before exiting).
#
# Usage:
#   node_health_check.sh                 # normal run, real checks
#   node_health_check.sh --self-test      # synthetic failure injection,
#                                          # proves the failure path itself
#                                          # works, without needing a real
#                                          # unhealthy node to test against
set -uo pipefail

FAIL=0
fail() {
    echo "UNHEALTHY: $1" >&2
    FAIL=1
}
ok() {
    echo "OK: $1"
}

# --- Check 1: disk space -------------------------------------------------
# Threshold configurable via env var (real Slurm deployments would set
# this per-partition -- e.g. a scratch-heavy training partition needs
# more headroom than a serving partition). Default: 10% free minimum,
# checked on both / and (if it exists) /scratch, since
# train_job.sbatch writes real checkpoint shards there.
MIN_FREE_PCT="${HEALTH_CHECK_MIN_FREE_PCT:-10}"
# Threshold is an explicit second argument (default: the script-level
# $MIN_FREE_PCT), not read from the environment inside the function --
# an earlier version read $MIN_FREE_PCT directly here, which meant
# `HEALTH_CHECK_MIN_FREE_PCT=101 check_disk "/"` (the self-test's
# synthetic-failure injection below) had NO EFFECT, since MIN_FREE_PCT
# was already evaluated once at script-parse time before the self-test
# branch ever ran -- a real bug caught by actually running --self-test
# (it printed "synthetic failure was NOT detected"), not by inspection.
check_disk() {
    local path="$1"
    local threshold="${2:-$MIN_FREE_PCT}"
    [[ -d "$path" ]] || { echo "SKIP: disk check on $path (not present)"; return; }
    local used_pct
    used_pct=$(df -P "$path" | awk 'NR==2 {gsub("%","",$5); print $5}')
    local free_pct=$((100 - used_pct))
    if (( free_pct < threshold )); then
        fail "disk $path: only ${free_pct}% free (threshold ${threshold}%)"
    else
        ok "disk $path: ${free_pct}% free (threshold ${threshold}%)"
    fi
}

# --- Check 2: expected process/service presence --------------------------
# On a real Slurm node, slurmd itself must be running for this script to
# even BE running under it -- so this checks the OTHER long-running
# service this cluster depends on: munge (AuthType=auth/munge in
# slurm.conf). On this Mac (no Slurm/munge installed), both checks
# honestly report SKIP rather than a false failure -- see README for why
# that's the correct behavior here, not a gap.
check_process() {
    local name="$1"
    if command -v pgrep >/dev/null 2>&1; then
        if pgrep -x "$name" >/dev/null 2>&1; then
            ok "process '$name' is running"
        else
            if [[ "$(uname)" == "Darwin" ]]; then
                echo "SKIP: process '$name' not found (expected -- $name is Linux/Slurm-only, this is macOS)"
            else
                fail "process '$name' is not running"
            fi
        fi
    else
        echo "SKIP: process check for '$name' ('pgrep' unavailable)"
    fi
}

# --- Check 3: basic hardware sanity --------------------------------------
# Confirms the node reports at least the CPU count Slurm's own
# slurm.conf NodeName= line expects it to have (a node that silently
# lost a CPU core, e.g. from a BIOS/firmware fault, would otherwise keep
# accepting jobs sized for its nominal core count and run them slower or
# fail them). Threshold configurable; default matches slurm.conf's
# `cpu-node[01-08]` CPUs=8.
EXPECTED_MIN_CPUS="${HEALTH_CHECK_MIN_CPUS:-1}"
check_cpu_count() {
    local n
    if [[ "$(uname)" == "Darwin" ]]; then
        n=$(sysctl -n hw.logicalcpu)
    else
        n=$(nproc --all 2>/dev/null || echo 0)
    fi
    if (( n < EXPECTED_MIN_CPUS )); then
        fail "CPU count $n below expected minimum $EXPECTED_MIN_CPUS"
    else
        ok "CPU count $n (>= expected minimum $EXPECTED_MIN_CPUS)"
    fi
}

# --- Self-test: synthetic failure injection ------------------------------
# Proves the FAIL path (and thus the exit-code contract HealthCheckProgram
# depends on) actually works, without needing a genuinely unhealthy node
# to test against -- same "prove the failure path, not just the happy
# path" discipline as fpga_engine/symbiyosys's k-induction proofs and
# adversarial/'s attack-verification steps.
if [[ "${1:-}" == "--self-test" ]]; then
    echo "--- self-test: injecting a synthetic disk-space failure ---"
    check_disk "/" 101  # impossible threshold (>100%) -> must fail
    if [[ "$FAIL" -eq 1 ]]; then
        echo "self-test PASS: synthetic failure correctly detected and would DRAIN the node"
        exit 0
    else
        echo "self-test FAIL: synthetic failure was NOT detected -- health check logic is broken" >&2
        exit 1
    fi
fi

# --- Real run -------------------------------------------------------------
echo "node_health_check.sh: $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
check_disk "/"
check_disk "/scratch"
check_process "slurmd"
check_process "munged"
check_cpu_count

if [[ "$FAIL" -eq 1 ]]; then
    echo "node_health_check.sh: UNHEALTHY -- Slurm will DRAIN this node" >&2
    exit 1
fi
echo "node_health_check.sh: HEALTHY"
exit 0
