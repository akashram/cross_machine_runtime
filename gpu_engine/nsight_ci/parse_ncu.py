#!/usr/bin/env python3
"""
parse_ncu.py — parse Nsight Compute CSV output, extract key metrics, output JSON.

Usage:
    ncu --set full --csv --log-file metrics.csv ./roofline_bench
    python3 parse_ncu.py metrics.csv --output metrics.json

ncu CSV format (CUDA 12.x):
    Row 0: "==PROF==" header (skipped)
    Row 1: column header row
        Columns: "ID", "Process ID", "Process Name", "Host Name",
                 "Kernel Name", "Kernel Time", "Context", "Stream",
                 then pairs: "<metric name>", "<metric unit>", "<metric value>", ...
    Row 2+: one row per profiled kernel launch

This parser handles the 3-column-per-metric layout and extracts the metrics
listed in METRICS_OF_INTEREST, producing a JSON array of per-kernel dicts.

Output JSON format:
    [
      {
        "kernel_name": "gemm_tiled...",
        "sm_util_pct": 98.4,
        "hbm_rd_gbs": 1423.1,
        "achieved_tflops": 76.2,
        "l1_hit_rate_pct": 44.3,
        ...
      },
      ...
    ]

Run on GPU hardware: verify column names with:
    ncu --query-metrics --csv | head -40
"""

import csv
import json
import sys
import re
import argparse
from pathlib import Path
from typing import Optional

# Mapping: ncu metric ID → friendly key in output JSON.
# These are the most diagnostically useful metrics.
METRICS_OF_INTEREST = {
    # SM utilization
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        "sm_util_pct",
    # Memory bandwidth (reads from HBM via L2)
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum":
        "hbm_read_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum":
        "hbm_write_bytes",
    # Compute (SASS-level)
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum":
        "fadd_insts",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum":
        "fmul_insts",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum":
        "ffma_insts",
    # Warp efficiency
    "smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_elapsed":
        "warp_eff_pct",
    # Memory coalescing
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
        "ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum":
        "ld_requests",
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active":
        "occupancy_pct",
    # Duration
    "gpu__time_duration.sum":
        "duration_ns",
    # L1 hit rate
    "l1tex__t_sector_hit_rate.pct":
        "l1_hit_rate_pct",
    # Tensor Core utilization
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed":
        "tensor_core_util_pct",
}

# Metric IDs that represent bytes and need unit-aware scaling
BYTE_METRICS = {"hbm_read_bytes", "hbm_write_bytes", "ld_sectors", "ld_requests"}


def _parse_float(s: str) -> Optional[float]:
    """Parse a metric value string, handling commas and empty strings."""
    if not s or s.strip() in ("", "-", "N/A"):
        return None
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def _find_metric_columns(headers: list[str]) -> dict[str, int]:
    """
    ncu CSV header layout: after 8 fixed columns, metrics appear as triples:
      <name_col>, <unit_col>, <value_col>
    where name_col contains the metric ID string.

    Returns dict: metric_id → value_column_index.
    """
    FIXED_COLS = 8  # ID, Process ID, Process Name, Host Name,
                     # Kernel Name, Kernel Time, Context, Stream
    metric_cols: dict[str, int] = {}
    i = FIXED_COLS
    while i + 2 < len(headers):
        metric_name = headers[i].strip()
        # value is at i+2 (skipping unit column)
        if metric_name:
            metric_cols[metric_name] = i + 2
        i += 3  # advance by 3 (name, unit, value)
    return metric_cols


def _derive_metrics(row_data: dict) -> dict:
    """Compute derived metrics from raw extracted values."""
    derived = {}

    # Achieved TFLOPS = (fadd + fmul + 2*ffma) / duration_s / 1e12
    fadd  = row_data.get("fadd_insts") or 0.0
    fmul  = row_data.get("fmul_insts") or 0.0
    ffma  = row_data.get("ffma_insts") or 0.0
    ns    = row_data.get("duration_ns")
    if ns and ns > 0:
        flops = fadd + fmul + 2.0 * ffma
        derived["achieved_tflops"] = flops / (ns / 1e9) / 1e12

    # Memory coalescing efficiency = requests / sectors (ideal = 1.0 means fully coalesced)
    ld_req  = row_data.get("ld_requests")
    ld_sec  = row_data.get("ld_sectors")
    if ld_req and ld_sec and ld_sec > 0:
        derived["coalescing_efficiency"] = ld_req / ld_sec

    # HBM bandwidth (GB/s)
    rd_b = row_data.get("hbm_read_bytes") or 0.0
    wr_b = row_data.get("hbm_write_bytes") or 0.0
    if ns and ns > 0:
        derived["hbm_bw_gbs"] = (rd_b + wr_b) / (ns / 1e9) / 1e9

    # Arithmetic intensity (FLOP/byte) for roofline
    if derived.get("achieved_tflops") and derived.get("hbm_bw_gbs"):
        total_bytes = rd_b + wr_b
        if total_bytes > 0 and ns and ns > 0:
            flops_d = (fadd + fmul + 2.0 * ffma)
            derived["arithmetic_intensity"] = flops_d / total_bytes

    return derived


def parse_ncu_csv(csv_path: str) -> list[dict]:
    """Parse ncu --csv output and return list of per-kernel metric dicts."""
    results = []
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return results

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Skip ncu preamble lines starting with "=="
    data_lines = [l for l in lines if not l.startswith("==")]
    if not data_lines:
        print("Warning: no data rows found (all lines were == headers)", file=sys.stderr)
        return results

    reader = csv.reader(data_lines)
    rows = list(reader)
    if len(rows) < 2:
        print("Warning: too few rows in CSV", file=sys.stderr)
        return results

    headers = rows[0]
    metric_cols = _find_metric_columns(headers)

    # Build reverse map: friendly_key → column_index
    extract_cols: dict[str, int] = {}
    for metric_id, friendly in METRICS_OF_INTEREST.items():
        if metric_id in metric_cols:
            extract_cols[friendly] = metric_cols[metric_id]
        # else: metric not profiled (depends on --set passed to ncu)

    KERNEL_NAME_COL = 4  # "Kernel Name" is always the 5th fixed column

    for row in rows[1:]:
        if len(row) < KERNEL_NAME_COL + 1:
            continue
        kernel_name = row[KERNEL_NAME_COL].strip()
        if not kernel_name:
            continue

        row_data: dict = {"kernel_name": kernel_name}

        for friendly, col_idx in extract_cols.items():
            if col_idx < len(row):
                val = _parse_float(row[col_idx])
                if val is not None:
                    row_data[friendly] = val

        # Derive composite metrics
        row_data.update(_derive_metrics(row_data))

        results.append(row_data)

    return results


def check_regressions(current: list[dict], baseline_path: str,
                      thresholds: dict[str, float]) -> list[str]:
    """
    Compare current metrics against a baseline JSON file.
    thresholds: {metric_key: max_allowed_regression_fraction}
    Returns list of regression messages (empty if clean).
    """
    baseline_path_obj = Path(baseline_path)
    if not baseline_path_obj.exists():
        return []

    with open(baseline_path_obj) as f:
        baseline = json.load(f)

    baseline_by_kernel = {k["kernel_name"]: k for k in baseline if "kernel_name" in k}
    regressions = []

    for cur in current:
        name = cur.get("kernel_name", "")
        if name not in baseline_by_kernel:
            continue
        base = baseline_by_kernel[name]
        for metric, max_regress in thresholds.items():
            cur_val  = cur.get(metric)
            base_val = base.get(metric)
            if cur_val is None or base_val is None or base_val == 0:
                continue
            change = (cur_val - base_val) / abs(base_val)
            # For "higher is better" metrics (TFLOPS, bandwidth), a drop is a regression.
            # For "lower is better" metrics (duration), an increase is a regression.
            is_regression = False
            if metric in ("achieved_tflops", "hbm_bw_gbs", "sm_util_pct",
                          "warp_eff_pct", "occupancy_pct", "coalescing_efficiency",
                          "tensor_core_util_pct", "l1_hit_rate_pct"):
                is_regression = (change < -max_regress)
            else:  # duration — increase is bad
                is_regression = (change > max_regress)

            if is_regression:
                regressions.append(
                    f"REGRESSION [{name}] {metric}: "
                    f"baseline={base_val:.4g} current={cur_val:.4g} "
                    f"({change*100:+.1f}%, threshold={max_regress*100:.0f}%)")

    return regressions


def main():
    parser = argparse.ArgumentParser(
        description="Parse ncu --csv output into structured JSON with derived metrics.")
    parser.add_argument("csv_file",
                        help="Path to ncu CSV output file")
    parser.add_argument("--output", "-o", default=None,
                        help="Write JSON to this file (default: stdout)")
    parser.add_argument("--baseline", default=None,
                        help="Compare against baseline JSON for regression detection")
    parser.add_argument("--regression-threshold", type=float, default=0.05,
                        help="Max allowed regression fraction (default: 0.05 = 5%%)")
    parser.add_argument("--summary", action="store_true",
                        help="Print a human-readable summary table")
    args = parser.parse_args()

    results = parse_ncu_csv(args.csv_file)

    if not results:
        print("No kernel data parsed.", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        print(f"\n{'Kernel':<40} {'TFLOPS':>8} {'HBM GB/s':>10} "
              f"{'SM%':>6} {'Occ%':>6} {'AI':>6}")
        print("-" * 78)
        for r in results:
            print(f"  {r['kernel_name'][:38]:<38} "
                  f"{r.get('achieved_tflops', 0):>8.2f} "
                  f"{r.get('hbm_bw_gbs', 0):>10.1f} "
                  f"{r.get('sm_util_pct', 0):>6.1f} "
                  f"{r.get('occupancy_pct', 0):>6.1f} "
                  f"{r.get('arithmetic_intensity', 0):>6.2f}")
        print()

    if args.baseline:
        # Default regression thresholds per metric
        thresholds = {
            "achieved_tflops":       args.regression_threshold,
            "hbm_bw_gbs":           args.regression_threshold,
            "sm_util_pct":           args.regression_threshold,
            "occupancy_pct":         args.regression_threshold,
            "coalescing_efficiency": args.regression_threshold,
            "duration_ns":           args.regression_threshold,
        }
        regressions = check_regressions(results, args.baseline, thresholds)
        if regressions:
            print("=== REGRESSIONS DETECTED ===", file=sys.stderr)
            for msg in regressions:
                print(f"  {msg}", file=sys.stderr)
            sys.exit(2)
        else:
            print("Regression check: PASS (no regressions vs baseline)")

    out = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {len(results)} kernel entries to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
