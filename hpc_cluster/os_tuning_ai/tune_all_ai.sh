#!/usr/bin/env bash
# tune_all_ai.sh -- orchestrates every live-tunable script in this
# directory in order, same "one orchestrator + individually-toggleable
# per-knob scripts, not a monolithic switch" pattern as
# cpu_engine/os_tuning/tune_all.sh -- so a future regression ("why did
# checkpoint throughput drop?") can be bisected to a specific knob
# instead of an opaque "tuning was applied" flag.
#
# tune_memlock.sh is deliberately NOT included here -- it edits a PAM
# config file that only takes effect on the NEXT login/job, not live, so
# it doesn't fit this script's "apply now" contract; run it separately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname)" != "Linux" ]]; then
    echo "os_tuning_ai/tune_all_ai.sh: tuning scripts are Linux-only."
    echo "Running check_current.sh instead to show what's read-able on $(uname):"
    echo ""
    bash "${SCRIPT_DIR}/check_current.sh"
    exit 0
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

echo "=== before ==="
bash "${SCRIPT_DIR}/check_current.sh"

echo ""
echo "=== applying ==="
bash "${SCRIPT_DIR}/tune_numa_balancing.sh"
bash "${SCRIPT_DIR}/tune_thp.sh"
bash "${SCRIPT_DIR}/tune_memory.sh"
bash "${SCRIPT_DIR}/tune_net_sysctls.sh"
bash "${SCRIPT_DIR}/tune_cgroup_v2.sh"

echo ""
echo "=== after ==="
bash "${SCRIPT_DIR}/check_current.sh"

echo ""
echo "NOTE: tune_memlock.sh was NOT run (PAM-based, next-login effect only)."
echo "Run it separately: sudo bash ${SCRIPT_DIR}/tune_memlock.sh"
