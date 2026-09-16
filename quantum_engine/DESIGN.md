# Quantum Computing & Hybrid Quantum-Classical Compute — Design

## 1. One simulator, ten steps, not ten simulators

Every gate-based step (2, 3, 4, 5, 6, 9) shares `state_vector.h`'s
`StateVector` class unmodified except for two additive primitives added
exactly when a later step's physics needed them:

- `renormalize()` (step 3) — needed the moment a noise channel's Kraus
  operator is sub-unitary and shrinks the state's norm.
- `measure_qubit()` (step 4) — needed the moment QEC's syndrome ancillas
  must be read out without collapsing the data qubits' logical
  superposition.

Both were added to `state_vector.h` directly rather than duplicated
locally in `noise_model.h`/`qec.h`, so every later step (5, 6, 9) that
also needs partial measurement or renormalization gets the same,
already-tested implementation. This mirrors `analog_engine`'s reuse
discipline (step 1's device model reused unmodified by every later step
that needs analog imprecision) applied to a genuinely different physics
domain.

## 2. Why Monte Carlo unraveling instead of a density-matrix simulator

`state_vector.h` is a PURE state vector (`O(2^n)` memory), a deliberate
choice matching this repo's "hand-roll the primitive first, zero new
dependencies" precedent (`foundation/proptest`, `observability/
opentelemetry`). A density-matrix representation (`O(4^n)` memory) is the
usual way to add mixed-state noise support, but it's a SEPARATE
representation from steps 1-2's pure-state work, not an extension of it.

Instead, `noise_model.h` (step 3) uses the Monte Carlo wavefunction
method (quantum trajectories/"quantum jumps", Dalibard, Castin & Mollmer
1992): every noise channel here has an exact Kraus decomposition, so
running many independent noisy PURE-state trajectories and averaging an
observable converges to exactly what a density-matrix simulation would
give. This keeps every step in the phase on the same underlying
representation — `qec.h` (step 4) directly reuses `apply_depolarizing`
from `noise_model.h` on the same pure-state objects `qec_trial_succeeds`
manipulates, with no representation mismatch to bridge.

## 3. Three different optimizers, chosen for three different reasons, not one default

- **VQE (step 5)**: exact analytic gradients via the parameter-shift rule
  (Mitarai et al. 2018). Every parameter appears in exactly ONE gate
  (`vqe_ansatz`'s per-qubit-per-layer `RY`), and `RY = exp(-i*theta*Y/2)`
  has a generator with eigenvalues `+-1` — the exact condition the
  two-point parameter-shift rule needs. Verified directly against
  central-difference finite differences before being trusted (see
  `vqe/README.md`).
- **QAOA (step 6)**: central-difference finite differences, NOT
  parameter-shift, because `gamma_l`/`beta_l` each appear in MULTIPLE
  gate occurrences sharing the same parameter (one `apply_zz` call per
  graph edge, one `rx` call per qubit) — the simple two-point rule
  doesn't directly apply to a whole shared-parameter layer as one gate.
  This is this repo's already-established pattern for exactly this
  situation (`sciml/ssm_layer`, `sciml/mup_scaling` both use
  finite-difference GD for the same underlying reason: no clean
  closed-form gradient available for the parameterization chosen).
- **PennyLane (step 7)**: the library's own automatic differentiation,
  deliberately NOT reimplementing either of the above — the entire point
  of this step is demonstrating the framework's real autodiff machinery,
  not this repo's hand-rolled version of it a third time.

## 4. Why CV/photonic primitives are hand-rolled, and why that's not a downgrade from Strawberry Fields

PLAN.md step 8 explicitly offers hand-rolling as the PREFERRED path (`the
Gaussian-state formalism is linear algebra on covariance matrices,
tractable without a new dependency`), not a fallback for when Strawberry
Fields isn't available. `cv_photonic.h` implements the real formalism
(Weedbrook et al. 2012) directly: a Gaussian state as `(mean, covariance)`,
gates as symplectic matrices, `is_symplectic()` checking the DEFINING
property (`S*Omega*S^T = Omega`) rather than trusting a gate's formula
because it looks like the textbook one. This mirrors step 6's `apply_zz`
discipline (verify the primitive standalone via a brute-force check
before composing it into a larger circuit) applied to an entirely
different mathematical structure — symplectic linear algebra instead of
unitary matrices on a Hilbert space.

The one thing intentionally left out of scope: full Gaussian Boson
Sampling requires computing output click-pattern PROBABILITIES via the
Hafnian of a submatrix of the interferometer's transformation matrix —
genuine #P-hard classical computation, and the actual source of GBS's
computational-hardness claim. `cv_photonic_test.cpp`'s GBS-style test
verifies the real, checkable invariant a passive linear-optical network
must satisfy (total photon number conservation through the
interferometer) rather than a fabricated click-probability number.

## 5. Real bugs, caught by checks that existed specifically to catch them

Three real implementation bugs surfaced this phase, each caught by a
check built BEFORE the bug was known to exist (not retrofitted after a
suspicious result):

1. **Grover's "bug" that wasn't a bug** (step 2): a wide brute-force
   search over Grover iteration counts found the closed-form optimal-
   iterations formula "wrong" at every `n` tested. Root cause: success
   probability is genuinely periodic (`sin^2((2k+1)*theta)`, period
   `pi/(2*theta)`), and the wide search spanned multiple periods,
   legitimately finding a later peak that happened to round to a
   marginally higher discrete probability than the first peak. Not a
   circuit bug — a test-design mistake, fixed by windowing the search
   around the formula's own prediction (the only iteration count anyone
   would actually use in practice, since continuing past the first peak
   wastes oracle queries).
2. **VQE's power-iteration normalization bug** (step 5): `std::sqrt(
   std::norm(inner(v, v)))` squared an already-real, already-squared
   quantity a second time (`std::norm()` on a complex SCALAR returns that
   scalar's own modulus squared; `inner(v,v)` is already the vector's
   squared norm). Caught by a hand-derivable closed-form sanity check
   (`H=Z0+Z1`, exact ground energy `-2.0`) BEFORE using power iteration to
   validate VQE against a harder Hamiltonian with no simple closed form —
   without that sanity check, the bug would have silently produced a
   wrong "ground truth" for every later comparison in the step.
3. **QAOA's learning-rate instability** (step 6): `lr=0.3` gradient
   ascent oscillated below its own random starting point on the first
   run. Confirmed as a step-size issue, not a gradient-sign or circuit
   bug, via a direct sweep (`{0.3, 0.1, 0.05, 0.02, 0.01}`) showing clean
   convergence at every `lr<=0.1` regardless of the exact value — kept as
   a documented finding in `qaoa/README.md` rather than a silently
   adjusted constant.

## 6. Consolidated qubit-indexing convention (stated once, used everywhere)

Every gate-based step uses the same convention, stated in full once in
`state_vector.h`'s header comment rather than re-derived per step: qubit
`q` is the `q`-th LEAST-significant bit of the basis-state index
(`|q_{n-1}...q_1 q_0>` maps to `sum_i q_i * 2^i`) — opposite of Nielsen &
Chuang's usual left-to-right convention, chosen because it makes
`index & (1<<qubit)` the direct bit test every gate implementation in
this phase relies on. `algorithms.h`'s QFT circuit needed an explicit
bit-reversal SWAP pass specifically because of this convention (the
textbook rotation-ladder circuit produces output in the OPPOSITE
qubit-index order); this is documented at the point of use rather than
assumed obvious.
