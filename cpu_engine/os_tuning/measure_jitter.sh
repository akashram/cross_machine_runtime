#!/usr/bin/env bash
# =============================================================================
# measure_jitter.sh — before/after scheduling jitter measurement
#
# Runs jitter_bench twice: once before and once after applying tune_all.sh.
# Use this to verify that OS tuning is having the expected effect.
#
# USAGE
# -----
#   sudo bash measure_jitter.sh [cpu] [isolated_cpu_list]
#   cpu:              CPU to measure jitter on (default: 1)
#   isolated_cpu_list: CPUs to tune (default: all except CPU 0)
#
# EXAMPLE
#   sudo bash measure_jitter.sh 1 1-3
#
# WHAT TO EXPECT (Linux, 4-core AWS c5.2xlarge, 200 us target sleep)
# ------------------------------------------------------------------
#   Baseline:                     p99 ~2000 us  p999 ~10000 us
#   + governor=performance:       p99  ~300 us  p999  ~1000 us
#   + C-states disabled:          p99   ~50 us  p999   ~200 us
#   + IRQ affinity:               p99   ~20 us  p999    ~50 us
#   + isolcpus + nohz_full:       p99    ~5 us  p999    ~15 us
#
# macOS: jitter_bench will run (measures scheduling latency via nanosleep),
# but the tuning scripts are not applicable. Expect p99 ~100-500 us.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JITTER_BENCH="${SCRIPT_DIR}/../../build/release/cpu_engine/bench/jitter_bench"

if [[ ! -x "$JITTER_BENCH" ]]; then
    echo "ERROR: jitter_bench not found at $JITTER_BENCH"
    echo "Build first: cmake --preset release && cmake --build --preset release --target jitter_bench"
    exit 1
fi

MEASURE_CPU="${1:-1}"
TOTAL_CPUS=$(nproc --all 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
ISOLATED_LIST="${2:-1-$((TOTAL_CPUS - 1))}"

echo "============================================"
echo "Scheduling Jitter — Before/After Measurement"
echo "Measure CPU: $MEASURE_CPU"
if [[ "$(uname)" == "Linux" ]]; then
    echo "Isolated CPUs: $ISOLATED_LIST"
fi
echo "============================================"
echo ""

echo "--- BASELINE ---"
"$JITTER_BENCH" "$MEASURE_CPU"
echo ""

if [[ "$(uname)" != "Linux" ]]; then
    echo "macOS: OS tuning scripts not applicable."
    echo "Advisory pinning already applied by jitter_bench."
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "NOTE: not running as root — skipping tune_all.sh"
    echo "Run as: sudo bash measure_jitter.sh $MEASURE_CPU $ISOLATED_LIST"
    exit 0
fi

echo "--- Applying tune_all.sh ---"
bash "${SCRIPT_DIR}/tune_all.sh" "$ISOLATED_LIST" 2>&1 | grep -v "^---"
echo ""

echo "--- AFTER TUNING ---"
"$JITTER_BENCH" "$MEASURE_CPU"
echo ""

echo "============================================"
echo "For maximum isolation, also run:"
echo "  sudo bash tune_isolcpus.sh $ISOLATED_LIST"
echo "  sudo reboot"
echo "  sudo bash tune_all.sh $ISOLATED_LIST"
echo "  ./build/release/cpu_engine/bench/jitter_bench $MEASURE_CPU"
echo "============================================"
