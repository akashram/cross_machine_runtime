# state_vector

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 20 step 1: a hand-rolled state-vector quantum simulator —
a real `2^n`-amplitude complex state vector, single-qubit gates (X, Y, Z,
H, RX, RY, RZ, phase) applied as structured 2x2 tensor contractions over
amplitude pairs, two-qubit controlled gates (CNOT, CZ, CPHASE, CCX/Toffoli)
applied as index-masked amplitude selection, and measurement/sampling via
the Born rule. Zero dependencies — same "hand-roll the primitive first"
precedent as `foundation/proptest` and `observability/opentelemetry`.

Every later step in this phase (algorithms, noise model, QEC, VQE, QAOA,
CV/photonic) builds on this simulator unmodified.

## Design

- Qubit `q` is the `q`-th least-significant bit of the basis-state index
  (`|q_{n-1}...q_1 q_0>` maps to `sum_i q_i * 2^i`) — the opposite of
  Nielsen & Chuang's usual left-to-right convention, chosen because it
  makes `index & (1<<qubit)` the direct bit test.
- `apply_1q` iterates once over the half of the amplitude vector with the
  target bit clear, updating each `(i, i|bit)` pair together — `O(2^n)`
  per single-qubit gate, zero allocation, no general matrix machinery for
  a case that's always exactly a 2x2 block.
- Controlled gates (`cnot`/`cz`/`cphase`/`ccx`) are index-masked: iterate
  all `2^n` basis indices, act only where the control bit(s) are set.
- `project_unnormalized(q, outcome)` zeros amplitudes inconsistent with a
  measurement outcome WITHOUT renormalizing — used by the teleportation
  test below to get an exact (renormalization-invariant) conditional
  amplitude ratio instead of a probabilistic sampling average.
- `measure_all` implements the actual Born-rule sampling + collapse a real
  simulator needs (used by later steps, e.g. QEC's syndrome measurement).

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  amps: |00>=0.7071 |01>=0.0000 |10>=0.0000 |11>=0.7071
PASS  H(q0);CNOT(0,1) produces the Bell state (|00>+|11>)/sqrt(2) exactly
  |000>=0.7071 |111>=0.7071 other-state probability mass=0.00e+00
PASS  H(q0);CNOT(0,1);CNOT(0,2) produces the 3-qubit GHZ state (|000>+|111>)/sqrt(2) exactly
  worst |sum|amp|^2 - 1| over 200 random gates on 4 qubits = 8.438e-15
PASS  norm^2 stays exactly 1 (to fp precision) after 200 random single/two-qubit unitary gates
  branch (m0=0,m1=0): a0=0.469686 a1=0.171449 ratio=0.365028 (expected 0.365028) branch-mass=0.250000
  branch (m0=0,m1=1): a0=0.469686 a1=0.171449 ratio=0.365028 (expected 0.365028) branch-mass=0.250000
  branch (m0=1,m1=0): a0=0.469686 a1=0.171449 ratio=0.365028 (expected 0.365028) branch-mass=0.250000
  branch (m0=1,m1=1): a0=0.469686 a1=0.171449 ratio=0.365028 (expected 0.365028) branch-mass=0.250000
PASS  teleportation: after X/Z correction, q2's conditional amplitude ratio matches q0's original cos(theta/2):sin(theta/2) ratio EXACTLY on all 4 measurement branches (not just on average)
PASS
```

## Findings

- **Unitarity holds to floating-point precision, not just approximately.**
  200 random single- and two-qubit gates on a 4-qubit register drift the
  total probability mass by at most `8.4e-15` from 1.0 — pure fp roundoff,
  confirming every gate implemented here really is unitary (a bug in, say,
  the RY sign convention would show up as a drifting norm, not just a
  wrong answer on a specific circuit).
- **Teleportation is checked exactly, not statistically.** Rather than
  running the circuit many times and averaging fidelity (which can hide a
  sign error that only breaks one of the four measurement branches), all
  four `(m0, m1)` outcome branches are projected out via
  `project_unnormalized` and their conditional amplitude ratio is compared
  directly against the source qubit's `cos(theta/2):sin(theta/2)` ratio —
  all four branches match to `1e-9`, and each branch's total probability
  mass is exactly `0.25` (as expected: 4 equally-likely Bell-measurement
  outcomes), confirming the X/Z correction rule is applied for the right
  reason, not tuned to pass on average.

## Platform notes

Single-threaded numerical code — no concurrency, so TSan is not
applicable (same as `analog_engine`'s steps). Builds and passes under
both `--preset debug` and `--preset asan` (ASan+UBSan), byte-identical
output, zero sanitizer findings.
