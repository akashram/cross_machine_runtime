#!/usr/bin/env python3
"""parse_power_report.py — turn a Vivado `report_power` text report into
structured JSON and enforce a power budget in CI.

synth.tcl (tcl_pipeline/) already runs `report_power -file reports/power.rpt`
for every bitstream build. This script is the CI-facing half: parse that
fixed-width text table into numbers, archive them as a JSON artifact next to
the .rpt, and fail the build if total on-chip power exceeds a budget — so a
kernel change that blows the power envelope is caught the same way a timing
regression is (tcl_pipeline/synth.tcl's WNS gate), not just left in a report
nobody reads.

TODO: run on F1. Untested against a real Vivado report_power output — the
parser below is written against Vivado's documented report_power text
format (Xilinx UG907), not a captured report, since no report has been
generated yet.
"""
import argparse
import json
import re
import sys

# Vivado's report_power text format has a "1. Summary" section with a table
# like:
#
# | Total On-Chip Power (W)  | 4.322     |
# | Dynamic (W)               | 3.912     |
# | Device Static (W)         | 0.410     |
# | Confidence Level           | Low       |
# | Design Nets Matched        | NA        |
#
# and a "1.1 On-Chip Components" breakdown table with a row per resource
# class (Clocks, Signals, Logic, BRAM, DSP, PS7/MMCM, I/O, ...).
SUMMARY_FIELDS = {
    "Total On-Chip Power (W)": "total_on_chip_power_w",
    "Dynamic (W)": "dynamic_power_w",
    "Device Static (W)": "static_power_w",
    "Junction Temperature (C)": "junction_temp_c",
    "Thermal Margin (C)": "thermal_margin_c",
}

COMPONENT_ROW_RE = re.compile(
    r"^\|\s*([A-Za-z0-9_ /]+?)\s*\|\s*([0-9]+\.[0-9]+)\s*\|"
)


def parse_report(text: str) -> dict:
    result: dict = {"summary": {}, "components": {}}

    in_components = False
    for line in text.splitlines():
        for label, key in SUMMARY_FIELDS.items():
            if label in line:
                m = re.search(r"\|\s*([0-9.\-]+)\s*\|", line.split(label, 1)[1])
                if m:
                    result["summary"][key] = float(m.group(1))

        if "On-Chip Components" in line:
            in_components = True
            continue
        if in_components:
            m = COMPONENT_ROW_RE.match(line)
            if m:
                name, watts = m.group(1).strip(), float(m.group(2))
                if name and name.lower() != "total":
                    result["components"][name] = watts

    if "total_on_chip_power_w" not in result["summary"]:
        raise ValueError("could not find 'Total On-Chip Power (W)' in report — "
                          "unexpected report_power format")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="path to Vivado report_power .rpt file")
    ap.add_argument("--out", help="write parsed JSON here (default: <report>.json)")
    ap.add_argument("--budget-w", type=float, default=None,
                     help="fail if Total On-Chip Power exceeds this many watts")
    args = ap.parse_args()

    with open(args.report) as f:
        text = f.read()

    parsed = parse_report(text)
    out_path = args.out or (args.report + ".json")
    with open(out_path, "w") as f:
        json.dump(parsed, f, indent=2)

    total = parsed["summary"]["total_on_chip_power_w"]
    print(f"parse_power_report: total on-chip power = {total:.3f} W "
          f"(parsed -> {out_path})")

    if args.budget_w is not None and total > args.budget_w:
        print(f"parse_power_report: FAIL — {total:.3f} W exceeds budget "
              f"of {args.budget_w:.3f} W", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
