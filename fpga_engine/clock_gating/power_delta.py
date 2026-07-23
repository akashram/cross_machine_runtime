#!/usr/bin/env python3
"""power_delta.py — compute measured dynamic power reduction between an
ungated and a gated-clock-conversion Vivado report_power run.

power_gating.tcl produces reports/power_ungated.rpt and
reports/power_gated.rpt from the same kernel (gated_mlp.cpp) built two
ways. This script parses both (same text-table format as
power_ci/parse_power_report.py, duplicated here rather than imported so
this step stays runnable standalone) and reports the dynamic power
reduction, so it can be compared directly against
clock_gating_model.cpp's predicted reduction for the duty cycle both
reports were generated at.

TODO: run on F1. Untested against real Vivado report_power output — see
the self-test at the bottom of this file, which validates the parsing
and delta math against two synthetic reports shaped like Vivado's
documented report_power format (Xilinx UG907), not captured hardware
output.
"""
import argparse
import re
import sys

SUMMARY_FIELDS = {
    "Total On-Chip Power (W)": "total_on_chip_power_w",
    "Dynamic (W)": "dynamic_power_w",
    "Device Static (W)": "static_power_w",
}


def parse_report(text: str) -> dict:
    summary: dict = {}
    for line in text.splitlines():
        for label, key in SUMMARY_FIELDS.items():
            if label in line:
                m = re.search(r"\|\s*([0-9.\-]+)\s*\|", line.split(label, 1)[1])
                if m:
                    summary[key] = float(m.group(1))

    if "dynamic_power_w" not in summary:
        raise ValueError("could not find 'Dynamic (W)' in report — "
                          "unexpected report_power format")
    return summary


def compute_delta(ungated: dict, gated: dict) -> dict:
    ungated_dyn = ungated["dynamic_power_w"]
    gated_dyn = gated["dynamic_power_w"]
    reduction_pct = (ungated_dyn - gated_dyn) / ungated_dyn * 100.0
    return {
        "ungated_dynamic_w": ungated_dyn,
        "gated_dynamic_w": gated_dyn,
        "reduction_pct": reduction_pct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ungated_report", help="path to the ungated build's report_power .rpt")
    ap.add_argument("gated_report", help="path to the -gated_clock_conversion build's report_power .rpt")
    ap.add_argument("--predicted-pct", type=float, default=None,
                     help="clock_gating_model.cpp's predicted reduction %%, for comparison")
    args = ap.parse_args()

    with open(args.ungated_report) as f:
        ungated = parse_report(f.read())
    with open(args.gated_report) as f:
        gated = parse_report(f.read())

    delta = compute_delta(ungated, gated)
    print(f"power_delta: ungated dynamic = {delta['ungated_dynamic_w']:.3f} W, "
          f"gated dynamic = {delta['gated_dynamic_w']:.3f} W, "
          f"reduction = {delta['reduction_pct']:+.1f}%")

    if args.predicted_pct is not None:
        diff = delta["reduction_pct"] - args.predicted_pct
        print(f"power_delta: model predicted {args.predicted_pct:+.1f}%, "
              f"measured {delta['reduction_pct']:+.1f}% (diff {diff:+.1f} pts)")

    return 0


def _self_test() -> None:
    """Synthetic reports shaped like Vivado's documented report_power
    format, exercising the parser + delta math without any real Vivado
    output — same rationale as power_ci/parse_power_report.py's test."""
    ungated_rpt = """
1. Summary
--------------------------------------------------------------------------------
| Total On-Chip Power (W)  | 4.322     |
| Dynamic (W)               | 3.912     |
| Device Static (W)         | 0.410     |
"""
    gated_rpt = """
1. Summary
--------------------------------------------------------------------------------
| Total On-Chip Power (W)  | 0.879     |
| Dynamic (W)               | 0.469     |
| Device Static (W)         | 0.410     |
"""
    ungated = parse_report(ungated_rpt)
    gated = parse_report(gated_rpt)
    delta = compute_delta(ungated, gated)

    assert abs(ungated["dynamic_power_w"] - 3.912) < 1e-9
    assert abs(gated["dynamic_power_w"] - 0.469) < 1e-9
    expected_reduction = (3.912 - 0.469) / 3.912 * 100.0
    assert abs(delta["reduction_pct"] - expected_reduction) < 1e-6
    print(f"power_delta._self_test: OK — synthetic reduction = {delta['reduction_pct']:.1f}%"
          f" (expected {expected_reduction:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
        sys.exit(0)
    sys.exit(main())
