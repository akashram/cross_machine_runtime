#!/usr/bin/env bash
# run_formal_ci.sh — re-runs every SymbiYosys formal proof in the repo
# and gates on all of them passing, the CI job PLAN.md Phase 10 step 5
# asks for ("formal verification runs on every FPGA RTL change").
#
# This isn't a new formal spec -- fpga_engine/symbiyosys/ already has two
# real, passing SymbiYosys proofs (step 21: axi_nodead.sby, dma_nooverlap.sby,
# against the RTL step 20's cocotb tests exercise). What was missing is
# the CI-integration piece: a script that re-runs both any time the RTL
# changes, gates a build/PR on the result, and reports a clear pass/fail
# summary instead of requiring someone to manually re-invoke `sby` and
# read its log.
#
# Unlike this Phase's other toolchain-gated steps, yosys/sby are actually
# installed here (~/oss-cad-suite, from Phase 7's step 21 -- see project
# memory) -- so this script is run for real below, not left unrun.
#
# Deliberately avoids bash 4+ features (associative arrays, etc.): macOS
# ships bash 3.2 by default (no homebrew bash installed here either), and
# some CI images run equally old shells -- two hardcoded proof entries
# below is clearer than an associative array anyway.
#
# Usage: ./run_formal_ci.sh [--force]
#   --force: re-run even if a proof's cached PASS file already reflects
#            the current RTL (checked via a source-file hash, not just
#            file presence -- a stale PASS from before an RTL edit must
#            not be trusted as still valid).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBY_DIR="$REPO_ROOT/fpga_engine/symbiyosys"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if ! command -v sby >/dev/null 2>&1; then
    echo "sby not found on PATH -- see fpga_engine/symbiyosys/README.md for the OSS CAD Suite install" >&2
    exit 1
fi

overall_status=0
printf "%-16s %-8s %-10s %s\n" "proof" "result" "cached?" "detail"

run_proof() {
    local name="$1" sby_file="$2" rtl_files="$3"
    local hash_file="$SBY_DIR/$name/.rtl_hash"
    local cur_hash
    cur_hash=$(cd "$SBY_DIR" && cat $rtl_files "$sby_file" | shasum -a 256 | cut -d' ' -f1)

    if [[ $FORCE -eq 0 && -f "$hash_file" && "$(cat "$hash_file")" == "$cur_hash" && -f "$SBY_DIR/$name/PASS" ]]; then
        printf "%-16s %-8s %-10s %s\n" "$name" "PASS" "yes" "RTL unchanged since last run"
        return 0
    fi

    if (cd "$SBY_DIR" && sby -f "$sby_file" >"/tmp/${name}_sby_ci.log" 2>&1); then
        echo "$cur_hash" > "$hash_file"
        printf "%-16s %-8s %-10s %s\n" "$name" "PASS" "no" "re-verified"
        return 0
    else
        printf "%-16s %-8s %-10s %s\n" "$name" "FAIL" "no" "see /tmp/${name}_sby_ci.log"
        return 1
    fi
}

run_proof axi_nodead axi_nodead.sby "../cocotb/axi_stream_passthrough.v axi_formal.v" || overall_status=1
run_proof dma_nooverlap dma_nooverlap.sby "../cocotb/dma_controller.v dma_formal.v" || overall_status=1

exit $overall_status
