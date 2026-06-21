#!/usr/bin/env python3
"""
sass_analyze.py — pattern analysis of cuobjdump --dump-sass output.

Usage:
    python3 sass_analyze.py <sass_dump.txt> [kernel_name_filter]

Checks for known inefficiency patterns in SASS and prints a summary table.

Pattern catalogue
-----------------
LDG.E vs LDG.E.128 / LDG.E.64
    Vectorized loads (128-bit) vs scalar (32-bit).
    More .128 means the compiler vectorised global loads (good).
    All-scalar loads on a bandwidth-bound kernel = missed vectorisation.

STL / LDL  (local memory store/load = register spill)
    Any STL/LDL means the compiler ran out of registers and spilled to
    local memory (slow, goes through L2/HBM).  High spill count ↔ low
    occupancy due to excessive register pressure.

NOP / NANOSLEEP
    NOP pads pipeline stalls.  A lot of NOPs means the compiler couldn't
    hide latency — consider software pipelining or reducing dependencies.

IMAD.WIDE / XMAD
    Integer multiply-accumulate used for address computation.  Present in
    every kernel that indexes arrays.  High counts mean lots of indexing
    arithmetic — consider simplifying the index expression.

S2R %r, %tid / %ctaid
    Thread/block index reads.  Expected at kernel entry.  If scattered
    throughout the kernel body, the compiler didn't hoist the index.

BAR.SYNC / __syncthreads
    Each BAR.SYNC is a barrier.  More than expected in a tiling loop = the
    compiler didn't merge barriers.
"""

import sys
import re
import collections

def parse_sass(path: str) -> dict[str, list[str]]:
    """Return {kernel_name: [instruction_line, ...]} from a sass dump."""
    kernels: dict[str, list[str]] = {}
    current: str | None = None
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*Function\s*:\s*(\S+)', line)
            if m:
                current = m.group(1)
                kernels[current] = []
                continue
            if current is not None:
                kernels[current].append(line.rstrip())
    return kernels


def count_pattern(lines: list[str], pattern: str) -> int:
    return sum(1 for l in lines if re.search(pattern, l))


PATTERNS = [
    ("LDG.E.128",    r'\bLDG\.E\.128\b',   "vectorised 128-bit global load (good)"),
    ("LDG.E.64",     r'\bLDG\.E\.64\b',    "vectorised 64-bit global load"),
    ("LDG.E (32)",   r'\bLDG\.E\b(?!\.)',   "scalar 32-bit global load"),
    ("STG.E.128",    r'\bSTG\.E\.128\b',   "vectorised 128-bit global store (good)"),
    ("STG.E (32)",   r'\bSTG\.E\b(?!\.)',   "scalar 32-bit global store"),
    ("STL",          r'\bSTL\b',            "register SPILL to local memory (bad)"),
    ("LDL",          r'\bLDL\b',            "register FILL from local memory (bad)"),
    ("NOP",          r'\bNOP\b',            "pipeline stall padding"),
    ("BAR.SYNC",     r'\bBAR\.SYNC\b',      "__syncthreads"),
    ("LDSM",         r'\bLDSM\b',           "shared-mem matrix load (Tensor Core feed)"),
    ("HMMA/WMMA",    r'\b(HMMA|WMMA)\b',   "Tensor Core multiply-accumulate"),
    ("IADD/IMAD",    r'\b(IADD|IMAD)\b',   "integer add/mul (indexing arithmetic)"),
]


def analyse_kernel(name: str, lines: list[str]) -> dict:
    total_instr = sum(1 for l in lines if re.search(r'\/\*[0-9a-f]+\*\/', l))
    counts = {}
    for label, pat, _ in PATTERNS:
        counts[label] = count_pattern(lines, pat)
    return {"total": total_instr, "counts": counts}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sass_path   = sys.argv[1]
    name_filter = sys.argv[2] if len(sys.argv) > 2 else ""

    kernels = parse_sass(sass_path)
    if name_filter:
        kernels = {k: v for k, v in kernels.items() if name_filter in k}

    if not kernels:
        print(f"No kernels found (filter={name_filter!r})")
        sys.exit(0)

    for kname, lines in kernels.items():
        info = analyse_kernel(kname, lines)
        total = info["total"]
        counts = info["counts"]

        print(f"\nKernel: {kname}  ({total} instructions)")
        print(f"{'Pattern':<18}  {'Count':>6}  {'% of total':>10}  Description")
        print("-" * 70)

        for label, _, desc in PATTERNS:
            c = counts[label]
            pct = (100.0 * c / total) if total > 0 else 0.0
            flag = ""
            if label in ("STL", "LDL") and c > 0:
                flag = "  ← SPILL WARNING"
            elif label == "LDG.E (32)" and counts.get("LDG.E.128", 0) == 0 and c > 10:
                flag = "  ← missed vectorisation?"
            print(f"  {label:<16}  {c:>6}  {pct:>9.1f}%  {desc}{flag}")

        # Vectorisation ratio
        ld_vec  = counts.get("LDG.E.128", 0) * 4 + counts.get("LDG.E.64", 0) * 2
        ld_tot  = ld_vec + counts.get("LDG.E (32)", 0)
        if ld_tot > 0:
            ratio = ld_vec / ld_tot * 100
            print(f"\n  Load vectorisation: {ratio:.0f}%  "
                  f"({ld_vec} vec-words of {ld_tot} total load words)")

        spills = counts.get("STL", 0) + counts.get("LDL", 0)
        if spills > 0:
            print(f"\n  *** {spills} register spill instructions — "
                  f"try __launch_bounds__ or reduce register pressure ***")

    print()


if __name__ == "__main__":
    main()
