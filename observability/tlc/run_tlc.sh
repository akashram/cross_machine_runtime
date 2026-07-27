#!/usr/bin/env bash
# run_tlc.sh — runs TLC against both existing TLA+ specs (networking/'s
# Raft and ring-all-reduce collective protocol).
#
# PLAN.md Phase 10 step 4: "run TLC on all TLA+ specs (Raft, all-reduce
# protocol), verify no violations under all interleavings."
#
# Copies each spec's .tla file (and its MC wrapper/.cfg, for Raft) into a
# scratch directory before invoking TLC, rather than relying on any
# particular TLC library-search-path flag to resolve the EXTENDS Raft in
# RaftMC.tla — guaranteed correct regardless of TLC version, at the cost
# of a trivial copy.
#
# Unrun here — no Java runtime on this Mac (see this repo's memory: the
# user asked to revisit this decision before hardware provisioning,
# same as Phase 8's JAX decision).
#
# Usage: TLA_JAR=/path/to/tla2tools.jar ./run_tlc.sh
# tla2tools.jar: https://github.com/tlaplus/tlaplus/releases

set -euo pipefail

TLA_JAR="${TLA_JAR:?set TLA_JAR=/path/to/tla2tools.jar (github.com/tlaplus/tlaplus/releases)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "== Raft (networking/tla_raft/Raft.tla, via RaftMC.tla) =="
cp "$REPO_ROOT/networking/tla_raft/Raft.tla" "$WORKDIR/"
cp "$SCRIPT_DIR/RaftMC.tla" "$SCRIPT_DIR/RaftMC.cfg" "$WORKDIR/"
( cd "$WORKDIR" && java -cp "$TLA_JAR" tlc2.TLC -config RaftMC.cfg RaftMC )

echo
echo "== Collective (networking/tla_collective/Collective.tla) =="
cp "$REPO_ROOT/networking/tla_collective/Collective.tla" "$WORKDIR/"
cp "$SCRIPT_DIR/Collective.cfg" "$WORKDIR/"
( cd "$WORKDIR" && java -cp "$TLA_JAR" tlc2.TLC -config Collective.cfg Collective )
