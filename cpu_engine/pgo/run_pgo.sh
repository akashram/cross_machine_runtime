#!/usr/bin/env bash
# PGO workflow — three steps in one script.
#
# Usage:  ./cpu_engine/pgo/run_pgo.sh
# Run from the repository root.
#
# Steps:
#   1. Instrument build  (-fprofile-instr-generate)
#   2. Training run      (generates pgo_train.profraw)
#   3. Merge profile     (llvm-profdata → build/pgo.profdata)
#   4. PGO-use build     (-fprofile-instr-use)
#   5. Compare baseline vs PGO

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ── 1. Instrument build ────────────────────────────────────────────────────
echo "=== Step 1: instrument build ==="
cmake --preset pgo-instrument -S "$ROOT" -B "$ROOT/build/pgo-instrument" -Wno-dev
cmake --build "$ROOT/build/pgo-instrument" --target pgo_train pgo_measure -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

# ── 2. Collect profile ─────────────────────────────────────────────────────
echo ""
echo "=== Step 2: collect profile ==="
PROFRAW="$ROOT/build/pgo-instrument/pgo_train.profraw"
LLVM_PROFILE_FILE="$PROFRAW" \
    "$ROOT/build/pgo-instrument/cpu_engine/pgo/pgo_train"

# ── 3. Merge profile ───────────────────────────────────────────────────────
echo ""
echo "=== Step 3: merge profile ==="
PROFDATA="$ROOT/build/pgo.profdata"
# llvm-profdata lives under Xcode CLT on macOS; find it portably
LLVM_PROFDATA="$(xcrun -f llvm-profdata 2>/dev/null || which llvm-profdata)"
"$LLVM_PROFDATA" merge "$PROFRAW" -output "$PROFDATA"
echo "Profile written to $PROFDATA"

# ── 4. PGO-use build ───────────────────────────────────────────────────────
echo ""
echo "=== Step 4: PGO-optimised build ==="
cmake --preset pgo-use -S "$ROOT" -B "$ROOT/build/pgo-use" -Wno-dev
cmake --build "$ROOT/build/pgo-use" --target pgo_measure -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

# ── 5. True baseline: plain release build (no instrumentation overhead) ───
echo ""
echo "=== Step 5: release baseline build ==="
cmake --preset release -S "$ROOT" -B "$ROOT/build/release" -Wno-dev
cmake --build "$ROOT/build/release" --target pgo_measure \
    -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

# ── 6. Compare release vs PGO ──────────────────────────────────────────────
echo ""
echo "=== Results ==="
echo ""
echo "--- Baseline (release, no PGO) ---"
"$ROOT/build/release/cpu_engine/pgo/pgo_measure"
echo ""
echo "--- PGO-optimised ---"
"$ROOT/build/pgo-use/cpu_engine/pgo/pgo_measure"
