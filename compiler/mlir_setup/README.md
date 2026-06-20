# MLIR Build Setup

**Status: STUB — requires Linux x86. Any AWS instance (c5.2xlarge recommended).**

## What this measures
Validates that LLVM/MLIR builds correctly from source, MLIR CMake integration
works, and the project can find all required dialects and tools.

## Results

TODO: run on Linux and fill in this table.

| Metric | Value |
|--------|-------|
| LLVM commit used | TODO |
| Build time (Ninja, 8 cores) | TODO |
| Disk usage (build dir) | TODO |
| `mlir-opt --version` | TODO |

## Hardware notes
- Required: Linux x86, ≥ 16GB RAM, ≥ 40GB disk
- Build preset: release (Linux)
- Recommended: c5.4xlarge (16 vCPU) to cut build time to ~30 min
