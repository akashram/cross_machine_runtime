#!/usr/bin/env python3
"""
parse_ncu.py — parse Nsight Compute CSV output, extract key metrics, output JSON.
Usage: python3 parse_ncu.py <metrics.csv> [--output metrics.json]

TODO: validate column names against actual ncu output on GPU hardware.
ncu --csv output format varies by version — verify on target machine.
"""

import csv
import json
import sys
import argparse

# Key metrics to extract (ncu metric IDs)
METRICS_OF_INTEREST = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_util_pct",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second": "hbm_bw_gbs",
    "smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_elapsed": "warp_eff_pct",
    "gpu__time_duration.sum": "duration_ns",
}

def parse_ncu_csv(csv_path: str) -> list[dict]:
    """Parse ncu CSV and return list of per-kernel metric dicts."""
    results = []
    # TODO: implement on GPU hardware — verify CSV column names with actual ncu output
    # ncu CSV has a complex format with multiple header rows
    # Column layout: "ID", "Process ID", "Process Name", "Host Name",
    #                "Kernel Name", "Kernel Time", "Context", "Stream",
    #                <metric_name>, <metric_unit>, <metric_value>, ...
    print(f"parse_ncu: STUB — validate CSV format on GPU hardware")
    print(f"  Would parse: {csv_path}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = parse_ncu_csv(args.csv_file)

    out = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {args.output}")
    else:
        print(out)

if __name__ == "__main__":
    main()
