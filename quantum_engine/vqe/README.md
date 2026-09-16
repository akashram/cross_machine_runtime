# vqe

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 20 step 5: the Variational Quantum Eigensolver — a
parameterized ansatz on `state_vector`'s simulator, driven by a classical
optimizer to minimize `<psi(theta)|H|psi(theta)>`, checked against exact
diagonalization for a small toy Hamiltonian.

## Design

- **Hamiltonian**: a 3-qubit transverse-field-Ising-like toy Hamiltonian,
  `H = h*(Z0+Z1+Z2) + J*(X0X1+X1X2)` with `h=1.0, J=0.5` — expressed as a
  sum of `PauliTerm`s (coefficient + one Pauli per qubit), the standard
  qubit-Hamiltonian representation.
- **Dense-matrix construction by basis-probing, not hand-derived
  Kronecker products**: `build_pauli_string_matrix` builds each Pauli
  string's `2^n x 2^n` matrix by applying this simulator's OWN `x()`/
  `y()`/`z()` gates to every computational basis vector and recording the
  result column-by-column — guaranteed consistent with how the simulator
  actually indexes qubits, with no separate hand-derived
  Kronecker-product convention that could silently disagree with it.
- **Exact ground-state energy**: power iteration on `shift*I - H` (shift
  chosen via `sum(|coeff|) + 1` to exceed `H`'s largest-magnitude
  eigenvalue) — the shifted operator's top eigenvalue corresponds to
  `H`'s SMALLEST eigenvalue, `energy = shift - top_eigenvalue`.
- **Ansatz**: a hardware-efficient ansatz, `depth` layers of (RY on every
  qubit; CNOT ladder), `n*depth` parameters.
- **Optimizer**: hand-rolled gradient descent using the **parameter-shift
  rule** (Mitarai, Negoro, Kitagawa & Fujii 2018) for EXACT analytic
  gradients (`d<H>/dtheta_i = 0.5*(E(theta_i+pi/2) - E(theta_i-pi/2))`,
  exact because `RY = exp(-i*theta*Y/2)` has a generator with eigenvalues
  `+-1`) — not ml/'s `LinearModel`/`BayesianOpt` classes, which are built
  around a Features/Labels supervised-learning shape that doesn't fit a
  scalar objective over an arbitrary real parameter vector with an exact
  closed-form gradient available. See `vqe.h`'s header comment for the
  full reasoning.

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  exact_ground_energy(Z0+Z1) = -2.000000 (closed-form answer: -2.0)
PASS  power-iteration exact_ground_energy matches the hand-derivable closed-form answer (-2.0) for H = Z0 + Z1
  theta[0]: parameter-shift=-0.504757 finite-diff=-0.504757
  ...
PASS  parameter-shift-rule gradient matches central-difference finite-difference gradient to 1e-4 at a random point
  exact ground energy = -3.124885
  VQE (depth=4) energy trace: start=2.891983 -> end=-3.124885 (300 iterations)
PASS  VQE energy decreases from its (randomly initialized) starting point
PASS  VQE (sufficiently expressive ansatz, depth=4) converges to within 0.01 of the exact ground-state energy
  best-of-8-restarts: depth=2 -> -3.061553 (gap=0.063333)  depth=4 -> -3.124885 (gap=0.000000)  exact=-3.124885
PASS  a deeper (more expressive) ansatz reaches a strictly better best-of-8-restarts energy than a shallow one on this Hamiltonian -- a real expressibility ceiling, not just an optimizer local-minimum artifact
PASS  the depth=4 ansatz's best-of-restarts energy is essentially exact (within 0.001)
PASS
```

## Findings

- **A real power-iteration bug, caught by the closed-form sanity check
  before it could silently corrupt every later comparison.** The first
  version computed the power-iteration vector norm as
  `std::sqrt(std::norm(inner(v, v)))`. `inner(v, v)` is already the real
  scalar `sum|v_i|^2` (the squared vector norm); `std::norm()` on a
  COMPLEX scalar returns that scalar's OWN modulus squared — so this
  computed `(sum|v_i|^2)`, squaring the already-squared norm a second
  time, not `sqrt(sum|v_i|^2)`. On the trivial `H = Z0+Z1` closed-form
  check (true answer: exactly `-2.0`), this produced `-0.601` — caught
  immediately because a hand-derivable ground truth existed to check
  against. Fixed by using `inner(v, v).real()` directly. Worth noting:
  this bug was NOT caught by the parameter-shift-vs-finite-difference
  gradient check (which passed on the first run, since it never depends
  on `exact_ground_energy`) — a reminder that independent checks only
  catch the bugs in the code path they actually exercise.
- **A real, checked expressibility ceiling, not an optimizer artifact.**
  With ansatz depth=1 or depth=2, VQE plateaus at the IDENTICAL energy
  (`-3.061553`, to 6 decimal places) across all 8 random restarts —
  `0.063` above the true ground energy — while depth=4 reaches the exact
  ground energy (gap `0.0000`) from every restart tried. Because every
  depth-2 restart lands on the exact same energy regardless of random
  initialization, this rules out "bad luck landing in a local minimum"
  as the explanation and confirms it's a genuine EXPRESSIBILITY limit —
  the depth-2 hardware-efficient ansatz's parameterized state manifold
  simply doesn't contain the true ground state for this Hamiltonian, no
  matter how the parameters are set. This is exactly the kind of
  "does the sophisticated setup actually reach the answer, checked, not
  assumed" finding this repo's other phases (e.g. Phase 12c's
  hyperparameter-optimization steps) apply throughout.

## Platform notes

Single-threaded numerical code, no concurrency — TSan not applicable.
Clean under both `--preset debug` and `--preset asan` (ASan+UBSan).
