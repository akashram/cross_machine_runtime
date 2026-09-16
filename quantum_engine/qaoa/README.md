# qaoa

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 20 step 6: QAOA (Quantum Approximate Optimization
Algorithm) applied to a small MaxCut instance, with an honest comparison
against a classical brute-force baseline — not an assumed
quantum-advantage narrative, matching Phase 12c's "does the sophisticated
method actually beat the simple baseline, checked" convention.

## Design

- **Instance**: `C5`, a 5-node cycle graph (edges `(0,1),(1,2),(2,3),
  (3,4),(4,0)`). Deliberately an ODD cycle — no bipartite 2-coloring
  exists, so the classical max cut is provably 4 of 5 edges, never all
  5 — a genuinely non-trivial instance, not a toy where the "obvious"
  answer is trivially all edges.
- **Classical baseline**: `classical_max_cut_bruteforce` exhaustively
  tries all `2^n` bipartitions — trivial at this size, and the direct
  ground truth QAOA is compared against.
- **Cost unitary**: `exp(-i*gamma*C)` for `C = sum_edges 0.5*(I -
  Z_i*Z_j)` reduces, since the identity term is an irrelevant global
  phase, to a per-edge `exp(i*(gamma/2)*Z_i*Z_j)` — implemented via
  `apply_zz`, the standard `CNOT(i,j); RZ(j,angle); CNOT(i,j)` identity
  (conjugating an RZ by CNOTs turns a single-qubit Z rotation into a
  two-qubit ZZ rotation). Checked directly against a brute-force
  per-basis-state diagonal-phase computation before being trusted inside
  the full circuit — the same "verify the primitive standalone before
  composing it" discipline as `qec`'s pairwise-parity argument.
- **Mixer unitary**: `exp(-i*beta*B)` for `B = sum_i X_i` factors into
  `RX(2*beta)` on every qubit independently, since each `X_i` acts on a
  different qubit.
- **Expectation**: `qaoa_expected_cut` computes `<C>` directly from the
  state vector's amplitudes (exact, since the simulator has full
  amplitude access — no shot-based sampling, a disclosed simplification
  real hardware wouldn't have; see Findings).
- **Optimizer**: gradient ASCENT via central-difference finite
  differences, not the exact parameter-shift rule `vqe.h` uses — `gamma`
  and `beta` each appear in MULTIPLE gate occurrences per layer (one
  `apply_zz` call per edge, one `rx` call per qubit) sharing the same
  parameter, so the simple two-point parameter-shift rule doesn't
  directly apply to a whole layer as one gate. Finite-difference GD is
  this repo's already-established pattern for exactly this situation
  (e.g. `sciml/ssm_layer`, `sciml/mup_scaling`).

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  basis=0: actual=(0.9151,-0.4032) expected=(0.9151,-0.4032)
  ...
PASS  apply_zz(angle) implements exp(-i*angle/2*Z_i*Z_j) exactly, checked against a brute-force per-basis-state phase computation
  classical brute-force max cut over all 32 assignments: 4 edges (assignment bits=5)
PASS  classical brute-force finds the max cut of C5 is exactly 4 (odd cycle: can't cut all 5)
  QAOA p=1: expected cut trace start=3.5431 -> end=3.7500 (classical max=4, approximation ratio=0.9375)
PASS  QAOA's expected cut value improves from its random starting point
PASS  QAOA's achieved expected cut is a genuine partial approximation of the classical optimum (strictly better than a random cut's ~0.5 ratio, never exceeding the true optimum)
  QAOA p=2: expected cut trace start=2.1279 -> end=4.0000 (classical max=4, approximation ratio=1.0000)
PASS  QAOA's expected cut value improves from its random starting point
PASS  QAOA's achieved expected cut is a genuine partial approximation of the classical optimum (strictly better than a random cut's ~0.5 ratio, never exceeding the true optimum)
PASS
```

## Findings

- **A real learning-rate sensitivity, caught before it could look like a
  QAOA correctness bug.** The first run used `lr=0.3` and FAILED both
  "improves from start" checks — `p=1` went `3.54 -> 1.36`, WORSE than
  its random start. A sweep across `lr in {0.3, 0.1, 0.05, 0.02, 0.01}`
  (kept as `qaoa_test.cpp`'s reasoning, not left as a silent constant
  change) showed `lr=0.3` overshoots repeatedly (min/max trace spanning
  `0.65` to `3.92` — wild oscillation, never settling), while `lr<=0.1`
  converges cleanly to the same answer regardless of the exact value
  chosen. This is expected for a bounded, trigonometric objective (the
  expected cut is a smooth periodic function of `gamma`/`beta` with a
  bounded range `[0, |E|]`) under plain gradient ascent with no
  step-size adaptation — not a QAOA-implementation bug, confirmed by
  `apply_zz`'s independent phase check passing throughout.
- **`p=2` reaches the EXACT classical optimum (`ratio=1.0000`) on this
  instance; `p=1` reaches `93.75%`.** This is the real, checked version
  of QAOA's central design claim — more layers (`p`) should approach the
  optimum more closely — demonstrated directly on an instance small
  enough to have a known exact answer, not assumed from the algorithm's
  general theory. Consistent with the honest, non-hyped framing PLAN.md
  asks for: this says nothing about quantum advantage (`n=5` is trivial
  classically), only that the algorithm's approximation-quality-vs-depth
  behavior is real and measured.
- **Disclosed simplification**: `qaoa_expected_cut` computes `<C>`
  exactly from full amplitude access, which no real QPU has — real
  hardware estimates `<Z_i*Z_j>` per edge via repeated shot-based
  sampling and averaging, introducing genuine statistical noise this
  exact-simulator version doesn't model. `noise_model.h`'s Monte Carlo
  machinery (step 3) could be composed with this step to add that
  realism; not done here since PLAN.md step 6 asks specifically for the
  classical-baseline comparison, not a noise study (step 3 already
  covers noise vs. fidelity).

## Platform notes

Single-threaded numerical code, no concurrency — TSan not applicable.
Clean under both `--preset debug` and `--preset asan` (ASan+UBSan).
