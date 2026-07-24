#!/usr/bin/env python3
"""parse_xadc_json.py — turn xadc_sensors.cpp's thermal.json / electrical.json
dumps into structured readings, and flag anything a thermal-aware router
(thermal_router/thermal_router.h, step 25) or a human would care about.

xadc_sensors.cpp (unrun — see its header) writes the raw JSON that
`xrt::device::get_info<xrt::info::device::thermal|electrical>()` returns.
This script is the portable half: no XRT dependency, runs anywhere Python
runs, so the parsing logic and its self-test can be exercised today the
same way power_ci/parse_power_report.py and clock_gating/power_delta.py
are — written against the *documented* schema shape, not a captured
report, until this runs on F1 for real.

Two things this deliberately does NOT do (left to step 25):
  - Decide a workload-allocation fraction from the temperature. That's
    thermal_router's ThermalPolicy — this script only extracts and
    flags readings, it doesn't own the throttle policy.
  - Measure response latency to a temperature change. That needs a real
    running router loop (step 25), not a one-shot report parse.
"""
import argparse
import json
import sys

# UltraScale+ VU9P nominal rail voltages (VU9P datasheet / AWS F1 shell
# power spec) used only to flag rails that look wrong, not to make a
# throttle decision — plain sanity-checking, same spirit as
# clock_gating's break-even constant: doesn't need to be exact to be
# useful, since a rail 10% off nominal is worth flagging at any
# reasonable tolerance choice.
NOMINAL_RAIL_VOLTS = {
    "vccint": 0.85,
    "vccbram": 0.85,
    "vccaux": 1.80,
    "12v_pex": 12.0,
    "12v_aux": 12.0,
    "3v3_pex": 3.30,
}
RAIL_TOLERANCE_FRAC = 0.05


def parse_thermal(thermal: dict) -> dict:
    """thermal.json shape: {"thermals": [{"description": str, "location_id": int,
    "temp_C": number}, ...]} — one entry per sensor the shell exposes
    (board-level PCB sensors plus the FPGA die itself)."""
    readings = {}
    for entry in thermal.get("thermals", []):
        desc = entry.get("description", "unknown")
        temp = entry.get("temp_C")
        if temp is not None:
            readings[desc] = float(temp)

    if "FPGA" not in readings:
        raise ValueError(
            "parse_thermal: no 'FPGA' entry in thermals[] — thermal_router needs "
            "die temperature specifically, not just board sensors. Either the "
            "shell reports it under a different description string (check the "
            "captured JSON once this runs on F1) or something is misconfigured."
        )
    return readings


def parse_electrical(electrical: dict) -> dict:
    """electrical.json shape: {"power_consumption_watts": str, "voltages":
    {rail_name: {"volts": str, "amps": str}, ...}}."""
    rails = {}
    for name, vals in electrical.get("voltages", {}).items():
        volts = vals.get("volts")
        if volts is not None:
            rails[name] = float(volts)
    return rails


def flag_rails(rails: dict) -> list:
    flags = []
    for name, volts in rails.items():
        nominal = NOMINAL_RAIL_VOLTS.get(name.lower())
        if nominal is None:
            continue
        deviation = abs(volts - nominal) / nominal
        if deviation > RAIL_TOLERANCE_FRAC:
            flags.append(
                f"{name}: {volts:.3f}V is {deviation * 100:.1f}% off nominal "
                f"{nominal:.3f}V (tolerance {RAIL_TOLERANCE_FRAC * 100:.0f}%)"
            )
    return flags


def analyze(thermal: dict, electrical: dict) -> dict:
    temps = parse_thermal(thermal)
    rails = parse_electrical(electrical)
    flags = flag_rails(rails)
    return {
        "die_temp_c": temps["FPGA"],
        "board_temps_c": {k: v for k, v in temps.items() if k != "FPGA"},
        "rail_volts": rails,
        "rail_flags": flags,
    }


def _self_test() -> None:
    # Synthetic reports shaped like the documented schema (see module
    # docstring) — proves the parsing/flagging logic, not real hardware
    # values. One rail (vccaux) deliberately pushed 8% off nominal so
    # flag_rails has something real to catch.
    thermal = {
        "thermals": [
            {"description": "PCB TOP FRONT", "location_id": 4, "temp_C": 33},
            {"description": "PCB TOP REAR", "location_id": 5, "temp_C": 32},
            {"description": "FPGA", "location_id": 1, "temp_C": 61},
        ]
    }
    electrical = {
        "power_consumption_watts": "42.7",
        "voltages": {
            "vccint": {"volts": "0.852", "amps": "24.1"},
            "vccbram": {"volts": "0.849", "amps": "1.8"},
            "vccaux": {"volts": "1.944", "amps": "0.6"},  # 8% high, should flag
            "12v_pex": {"volts": "12.03", "amps": "3.1"},
        },
    }

    result = analyze(thermal, electrical)
    assert result["die_temp_c"] == 61, result
    assert result["board_temps_c"] == {"PCB TOP FRONT": 33.0, "PCB TOP REAR": 32.0}, result
    assert len(result["rail_flags"]) == 1 and "vccaux" in result["rail_flags"][0], result

    print("parse_xadc_json._self_test: OK")
    print(f"  die_temp_c = {result['die_temp_c']}")
    print(f"  rail_flags = {result['rail_flags']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("thermal_json", nargs="?", help="path to xadc_sensors.cpp's thermal.json")
    ap.add_argument("electrical_json", nargs="?", help="path to xadc_sensors.cpp's electrical.json")
    ap.add_argument("--self-test", action="store_true", help="run against synthetic reports, no files needed")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if not args.thermal_json or not args.electrical_json:
        ap.error("thermal_json and electrical_json are required unless --self-test")

    with open(args.thermal_json) as f:
        thermal = json.load(f)
    with open(args.electrical_json) as f:
        electrical = json.load(f)

    result = analyze(thermal, electrical)
    print(json.dumps(result, indent=2))

    if result["rail_flags"]:
        print(f"parse_xadc_json: {len(result['rail_flags'])} rail(s) out of tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
