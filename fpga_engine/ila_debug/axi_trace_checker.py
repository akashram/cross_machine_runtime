#!/usr/bin/env python3
"""axi_trace_checker.py — check a captured AXI4-Stream cycle trace against
the two handshake rules an ILA session is normally used to eyeball by hand
(ARM AMBA AXI4-Stream Protocol spec, section on the VALID/READY handshake):

  1. VALID-hold: once TVALID is asserted, it must stay asserted every
     cycle until a transfer occurs (TREADY seen high in the same cycle).
     A source is not allowed to withdraw an offered word just because the
     sink hasn't accepted it yet.
  2. DATA-stability: while a source is asserting TVALID and waiting for
     TREADY (no transfer yet), the payload (TDATA/TLAST) must not change
     cycle to cycle. Whatever word was offered has to still be the word
     on the bus when the sink finally accepts it.

PLAN.md step 19 asks for an ILA debug session on `axi_stream/axi_passthrough.cpp`
that either captures a real bug or demonstrates probing a protocol. There's
no F1 instance here to run a real capture (see `ila_probes.tcl`'s header),
so this is the demonstrate-probing-a-protocol half: a portable trace
checker that does mechanically, on a CSV, exactly what you'd otherwise do
by eye in Vivado's waveform viewer after a real `run_hw_ila` capture —
and its self-test (below) runs a synthetic buggy trace shaped like a real,
common mistake (a hand-written AXI-Stream source with a free-running
counter feeding TDATA, ignoring backpressure) through it to prove the two
rules actually catch something.

Input CSV schema (this script's own canonical schema, not Vivado's
`write_hw_ila_data -csv_file` layout — see ila_probes.tcl's header for why
those two aren't the same thing yet): one row per sampled clock cycle,
columns `cycle,tvalid,tready,tdata,tlast`. Run once per AXI4-Stream port
captured (e.g. once for in_stream, once for out_stream).
"""
import argparse
import csv
import sys


def check_trace(rows: list) -> list:
    """rows: list of dicts with int cycle/tvalid/tready/tdata/tlast, sorted
    by cycle. Returns a list of violation dicts."""
    violations = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        was_waiting = prev["tvalid"] == 1 and prev["tready"] == 0
        if not was_waiting:
            continue

        if cur["tvalid"] != 1:
            violations.append({
                "cycle": cur["cycle"],
                "rule": "valid-hold",
                "detail": f"TVALID dropped at cycle {cur['cycle']} while cycle "
                          f"{prev['cycle']} was still waiting for TREADY (no transfer occurred)",
            })
        if cur["tdata"] != prev["tdata"] or cur["tlast"] != prev["tlast"]:
            violations.append({
                "cycle": cur["cycle"],
                "rule": "data-stability",
                "detail": f"payload changed at cycle {cur['cycle']} "
                          f"(tdata {prev['tdata']}->{cur['tdata']}, tlast {prev['tlast']}->{cur['tlast']}) "
                          f"while cycle {prev['cycle']} was still waiting for TREADY",
            })
    return violations


def load_csv(path: str) -> list:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "cycle": int(row["cycle"]),
                "tvalid": int(row["tvalid"]),
                "tready": int(row["tready"]),
                "tdata": int(row["tdata"], 0),
                "tlast": int(row["tlast"]),
            })
    rows.sort(key=lambda r: r["cycle"])
    return rows


def _clean_trace() -> list:
    # 3-word transfer (0xA0, 0xA1, 0xA2, last word marked TLAST) with the
    # sink deasserting TREADY for two cycles after the first word — a
    # correctly-behaved source holds TDATA/TVALID steady across that wait.
    return [
        {"cycle": 0, "tvalid": 1, "tready": 1, "tdata": 0xA0, "tlast": 0},
        {"cycle": 1, "tvalid": 1, "tready": 0, "tdata": 0xA1, "tlast": 0},
        {"cycle": 2, "tvalid": 1, "tready": 0, "tdata": 0xA1, "tlast": 0},
        {"cycle": 3, "tvalid": 1, "tready": 1, "tdata": 0xA1, "tlast": 0},
        {"cycle": 4, "tvalid": 1, "tready": 1, "tdata": 0xA2, "tlast": 1},
        {"cycle": 5, "tvalid": 0, "tready": 0, "tdata": 0x00, "tlast": 0},
    ]


def _buggy_trace() -> list:
    # A hand-written (non-HLS) AXI-Stream source with a free-running
    # counter feeding TDATA and no backpressure handling: it keeps
    # incrementing TDATA every cycle regardless of TREADY, and drops
    # TVALID for a cycle while a word was still outstanding. Both are
    # real mistakes people make writing AXI-Stream logic by hand instead
    # of trusting an HLS-generated adapter.
    return [
        {"cycle": 0, "tvalid": 1, "tready": 1, "tdata": 0x00, "tlast": 0},
        {"cycle": 1, "tvalid": 1, "tready": 0, "tdata": 0x01, "tlast": 0},
        {"cycle": 2, "tvalid": 0, "tready": 0, "tdata": 0x02, "tlast": 0},
        {"cycle": 3, "tvalid": 1, "tready": 1, "tdata": 0x03, "tlast": 1},
        {"cycle": 4, "tvalid": 0, "tready": 0, "tdata": 0x00, "tlast": 0},
    ]


def _self_test() -> None:
    clean_violations = check_trace(_clean_trace())
    assert clean_violations == [], f"expected no violations on clean trace, got {clean_violations}"

    buggy_violations = check_trace(_buggy_trace())
    assert len(buggy_violations) == 2, f"expected 2 violations on buggy trace, got {buggy_violations}"
    assert {v["rule"] for v in buggy_violations} == {"valid-hold", "data-stability"}
    assert all(v["cycle"] == 2 for v in buggy_violations)

    print("axi_trace_checker._self_test: OK")
    print("  clean trace (axi_passthrough-shaped, backpressure for 2 cycles): 0 violations")
    print(f"  buggy trace (free-running-counter source, ignores backpressure): {len(buggy_violations)} violations")
    for v in buggy_violations:
        print(f"    cycle {v['cycle']} [{v['rule']}]: {v['detail']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_csv", nargs="?", help="path to a captured trace (cycle,tvalid,tready,tdata,tlast)")
    ap.add_argument("--self-test", action="store_true", help="run against synthetic clean/buggy traces, no file needed")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if not args.trace_csv:
        ap.error("trace_csv is required unless --self-test")

    rows = load_csv(args.trace_csv)
    violations = check_trace(rows)

    if not violations:
        print(f"axi_trace_checker: {args.trace_csv} — {len(rows)} cycles, no protocol violations")
        return 0

    print(f"axi_trace_checker: {args.trace_csv} — {len(violations)} violation(s):", file=sys.stderr)
    for v in violations:
        print(f"  cycle {v['cycle']} [{v['rule']}]: {v['detail']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
