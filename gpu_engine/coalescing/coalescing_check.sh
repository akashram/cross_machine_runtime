#!/usr/bin/env bash
# coalescing_check.sh — validate memory coalescing ratio for CUDA kernels via ncu
#
# Usage:
#   ./coalescing_check.sh <binary> [kernel_name] [threshold=0.90] [ideal_ratio=4.0]
#
# kernel_name: empty string or omitted → profile all kernels in the binary.
# threshold:   minimum coalescing efficiency (0.0–1.0). Kernels with
#              sectors/request > ideal_ratio/threshold are flagged as FAIL.
# ideal_ratio: sectors/request for a perfectly coalesced fp32 kernel on this GPU.
#              Default 4.0 (32 threads × 4B = 128B = 4 × 32B sectors per warp request).
#              Run coalescing_test first and set this to coalesced_kernel's measured ratio.
#
# Exit 0: all kernels pass. Exit 1: one or more kernels fail.

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [kernel_name] [threshold] [ideal_ratio]}"
KERNEL="${2:-}"
THRESHOLD="${3:-0.90}"
IDEAL_RATIO="${4:-4.0}"

METRICS="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"

CSV_OUT=$(mktemp /tmp/ncu_coalesce_XXXXXX.csv)
trap "rm -f ${CSV_OUT}" EXIT

NCU_CMD=(ncu --metrics "${METRICS}" --csv)
[[ -n "${KERNEL}" ]] && NCU_CMD+=(--kernel-name "${KERNEL}")
NCU_CMD+=("${BINARY}")

echo "[coalescing_check] Running: ${NCU_CMD[*]}"
"${NCU_CMD[@]}" > "${CSV_OUT}" 2>/dev/null

python3 - "${CSV_OUT}" "${THRESHOLD}" "${IDEAL_RATIO}" <<'PYEOF'
import sys, csv, collections

csv_path    = sys.argv[1]
threshold   = float(sys.argv[2])
ideal_ratio = float(sys.argv[3])
max_ratio   = ideal_ratio / threshold

ld_sectors  = collections.defaultdict(float)
ld_requests = collections.defaultdict(float)
st_sectors  = collections.defaultdict(float)
st_requests = collections.defaultdict(float)

with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
        kname = row.get("Kernel Name", "").strip()
        mname = row.get("Metric Name",  "").strip()
        mval  = row.get("Metric Value", "").strip().replace(",", "")
        if not kname or not mval:
            continue
        try:
            v = float(mval)
        except ValueError:
            continue
        if   mname == "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":  ld_sectors[kname]  += v
        elif mname == "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": ld_requests[kname] += v
        elif mname == "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum":  st_sectors[kname]  += v
        elif mname == "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": st_requests[kname] += v

kernels = sorted(set(ld_sectors) | set(st_sectors))
if not kernels:
    print("[coalescing_check] ERROR: no coalescing metrics found in CSV output")
    sys.exit(2)

hdr = f"{'Kernel':<50} {'LD sec/req':>12} {'ST sec/req':>12} {'Efficiency':>12} {'Pass':>6}"
print(hdr)
print("-" * len(hdr))

failed = []
for kname in kernels:
    ld_r = ld_sectors[kname] / ld_requests[kname] if ld_requests[kname] > 0 else 0.0
    st_r = st_sectors[kname] / st_requests[kname] if st_requests[kname] > 0 else 0.0
    # Efficiency = ideal / actual (1.0 = perfect, <1 = uncoalesced waste)
    ld_eff = (ideal_ratio / ld_r) if ld_r > 0 else 1.0
    ld_ok  = ld_r <= max_ratio
    st_ok  = st_requests[kname] == 0 or st_r <= max_ratio
    ok     = ld_ok and st_ok
    status = "PASS" if ok else "FAIL"
    print(f"{kname:<50} {ld_r:>12.3f} {st_r:>12.3f} {ld_eff:>11.1%} {status:>6}")
    if not ok:
        failed.append(kname)

print()
print(f"Threshold: {threshold*100:.0f}% efficiency  |  ideal ratio: {ideal_ratio}  |  max ratio: {max_ratio:.2f}")
if failed:
    print(f"\n[coalescing_check] FAIL: {len(failed)} kernel(s) below threshold:")
    for k in failed:
        print(f"  - {k}")
    sys.exit(1)
else:
    print(f"\n[coalescing_check] PASS: all {len(kernels)} kernel(s) meet coalescing threshold")
    sys.exit(0)
PYEOF
