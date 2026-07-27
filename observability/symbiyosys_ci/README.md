# symbiyosys_ci

**Status: code-complete AND locally run — yosys/sby are actually installed
on this Mac (OSS CAD Suite, from Phase 7 step 21), so unlike most of this
phase's toolchain-gated steps, this one runs for real.**

## What this measures

PLAN.md Phase 10 step 5: formal verification runs on every FPGA RTL
change.

## Design

`fpga_engine/symbiyosys/` already has two real, passing SymbiYosys proofs
(Phase 7 step 21: `axi_nodead.sby`, `dma_nooverlap.sby`) against the RTL
Phase 7 step 20's cocotb tests exercise. What this step adds is the
CI-integration piece PLAN.md actually asks for — re-running both any time
the RTL changes, gating a build on the result, one clear summary instead
of manually re-invoking `sby` and reading its log:

- `run_formal_ci.sh` hashes each proof's RTL + `.sby` sources
  (`shasum -a 256`) and skips re-running `sby` if the hash matches the
  last successful run's cached hash — a real change-detection gate, not
  just "run every time" (which would be correct but slow this down as
  more proofs accumulate).
- `--force` bypasses the cache, for a from-scratch CI run.
- Deliberately avoids bash 4+ features (associative arrays, etc.):
  macOS ships bash 3.2 by default and this Mac has no homebrew bash
  installed either, so a script relying on bash 4 wouldn't even run
  here — the first version of this script hit exactly that (`declare -A`
  failing with "unbound variable" under bash 3.2), fixed by using two
  explicit calls to a small `run_proof` function instead of an
  associative array (also just clearer for two proofs).
- Exit code is nonzero if any proof fails — the actual CI-gating
  behavior, not just a report.

## Results (captured 2026-07-27, this Mac, `~/oss-cad-suite` on PATH)

First run (`--force`, bypassing the cache to force real re-verification):

```
proof            result   cached?    detail
axi_nodead       PASS     no         re-verified
dma_nooverlap    PASS     no         re-verified
```

Second run (no `--force`, RTL unchanged — exercises the cache path):

```
proof            result   cached?    detail
axi_nodead       PASS     yes        RTL unchanged since last run
dma_nooverlap    PASS     yes        RTL unchanged since last run
```
Exit code: 0 in both cases.

## Hardware notes
None — `sby`/`yosys`/`z3` are already installed locally
(`~/oss-cad-suite`, per this repo's memory of the Phase 7 step 21
install). A CI runner would need the same OSS CAD Suite install (or the
paid Tabby CAD / commercial Yosys+Verific build, if SVA syntax beyond
what the free suite parses is ever needed — see
`fpga_engine/symbiyosys/README.md`'s note on that).
