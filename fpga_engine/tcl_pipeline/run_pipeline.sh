#!/usr/bin/env bash
# run_pipeline.sh — CI entry point wrapping synth.tcl for one kernel.
#
# Every kernel directory under fpga_engine/ that has real RTL/IP to build
# calls this the same way, so CI (or a developer) doesn't need to know
# Vivado's TCL argument-passing conventions:
#
#   ./run_pipeline.sh <kernel_dir> <top_module>
#
# TODO: run on F1 — depends on synth.tcl, which is itself unrun (see its
# header). This wrapper's own logic (arg handling, exit-code propagation,
# report archiving) needs no hardware and could be dry-run-tested with a
# stub 'vivado' on PATH, but that hasn't been done yet either.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <kernel_dir> <top_module>" >&2
    exit 1
fi

KERNEL_DIR="$1"
TOP="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$KERNEL_DIR/build_${TOP}"

vivado -mode batch -source "$SCRIPT_DIR/synth.tcl" -tclargs \
    -top "$TOP" \
    -ip_dir "$KERNEL_DIR/hls_ip" \
    -xdc "$KERNEL_DIR/constraints/${TOP}.xdc" \
    -outdir "$OUTDIR"
status=$?

if [[ $status -ne 0 ]]; then
    echo "run_pipeline.sh: build of '$TOP' FAILED (exit $status) — see $OUTDIR/reports/" >&2
    exit $status
fi

echo "run_pipeline.sh: build of '$TOP' succeeded — bitstream at $OUTDIR/${TOP}.bit"
