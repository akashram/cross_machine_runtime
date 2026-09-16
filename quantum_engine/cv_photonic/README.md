# cv_photonic

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 20 step 8: continuous-variable (CV) / photonic quantum
computing primitives — qumodes, Gaussian states, squeezing/displacement/
beamsplitter gates, homodyne measurement, and a small Gaussian-Boson-
Sampling-style circuit. This is Xanadu's specific hardware approach and
a genuinely different formalism from steps 1-7's qubit state vector, not
"qubits but optical" — chosen to be hand-rolled (per PLAN.md step 8's
explicit preference) rather than built on Strawberry Fields, which the
user declined installing this session alongside PennyLane.

## Design

- **Formalism**: a Gaussian state is a mean vector + covariance matrix
  over quadratures `(x_1,p_1,...,x_n,p_n)`, `[x,p]=i` (hbar=1), vacuum
  covariance `= I` — Weedbrook et al. 2012 (*Gaussian Quantum
  Information*, Rev. Mod. Phys. 84), a self-consistent convention picked
  explicitly (see `cv_photonic.h`'s header comment on why the specific
  hbar convention doesn't affect any check in this step).
- **Gates as symplectic transformations**: `squeeze` and `beamsplitter`
  are both built from a small local matrix (`2x2` / `4x4`) embedded into
  the full `2n x 2n` identity and applied as `cov -> S cov S^T, mean ->
  S mean`. `is_symplectic` checks the DEFINING property directly
  (`S*Omega*S^T = Omega`) rather than assuming the formulas are correct
  because they look like the textbook ones — the same "verify the
  primitive standalone" discipline as `qaoa`'s `apply_zz` check.
- **Displacement** is a pure mean-vector translation (not a matrix
  transformation) — covariance unchanged.
- **Mean photon number**: `n_bar = (Vx+Vp)/4 - 1/2 + (dx^2+dp^2)/2`,
  checked against the closed-form `sinh^2(r)` result for squeezed vacuum
  specifically (a standard, independently-derivable formula).
- **Homodyne measurement**: samples from the measured quadrature's
  marginal Gaussian (mean, variance read directly off the state).
  **Disclosed simplification**: this does NOT implement the full Gaussian
  conditional-state collapse of the OTHER modes after measurement (the
  standard Schur-complement formula from Weedbrook et al.) — only the
  measured quadrature's own marginal statistics, sufficient to
  demonstrate the measurement primitive itself.
- **GBS-style circuit**: multiple squeezed modes through a small
  interferometer built from `beamsplitter` calls. **Disclosed scope
  limitation**: full Gaussian Boson Sampling (Hamilton et al. 2017)
  requires computing output click-pattern PROBABILITIES via the Hafnian
  of a submatrix of the interferometer's transformation matrix — genuine
  #P-hard classical computation, and the actual source of GBS's
  computational-hardness claim. That's out of scope here; what IS built
  and checked is the state-preparation + linear-optics half (squeezing +
  a real interferometer), verified against the real physical invariant a
  passive, lossless linear-optical network must satisfy: total photon
  number is conserved through it exactly.

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  r=0.60: Vx=0.301194 (expected 0.301194)  Vp=3.320117 (expected 3.320117)  Vx*Vp=1.000000000
PASS  single-mode squeezing produces exact closed-form Vx=e^-2r, Vp=e^2r and stays minimum-uncertainty (Vx*Vp=1 exactly)
  r=0.2: mean_photon_number=0.040536 closed-form sinh^2(r)=0.040536
  r=0.5: mean_photon_number=0.271540 closed-form sinh^2(r)=0.271540
  r=1.0: mean_photon_number=1.381098 closed-form sinh^2(r)=1.381098
  r=1.5: mean_photon_number=4.533831 closed-form sinh^2(r)=4.533831
PASS  squeezed-vacuum mean photon number matches the closed-form sinh^2(r) result exactly, for r=0.2..1.5
PASS  squeeze_local_matrix(r) and beamsplitter_local_matrix(theta) satisfy S*Omega*S^T=Omega exactly, for several r/theta values
  total photon number before beamsplitter=1.222454061 after=1.222454061
PASS  a beamsplitter (a passive, lossless linear-optical element) conserves TOTAL photon number exactly
  3-mode GBS-style circuit: total photon number before interferometer=0.939722159 (closed-form sum of sinh^2=0.939722159) after=0.939722159
PASS  a small GBS-style circuit (3 squeezed modes through a 2-beamsplitter interferometer) conserves total photon number through the interferometer, matching the closed-form pre-interferometer total exactly
  homodyne x-samples (n=200000): empirical mean=1.1996 (true=1.2000) empirical var=0.3683 (true=0.3679)
PASS  homodyne sampling's empirical mean/variance over 200k trials match the state's own marginal mean/variance closely
PASS
```

## Findings

- **No implementation bugs found** — every test passed on the first
  run, including the closed-form checks (`Vx=e^-2r` to 6 decimal places,
  `n_bar=sinh^2(r)` to 6 decimal places across 4 values of `r`) and the
  symplectic-property check. The main risk area (matrix-embedding index
  bookkeeping in `apply_symplectic`, and the sign convention in
  `beamsplitter_local_matrix`) turned out correct on the first attempt,
  plausibly because `is_symplectic`'s direct algebraic check and the
  closed-form comparisons leave very little room for a sign or ordering
  error to hide (an error there would almost certainly break the exact
  `Vx*Vp=1` or `sinh^2(r)` match, not just degrade it slightly).
- **Total photon number conservation is exact to `1e-9`**, both for a
  single beamsplitter and for the full 3-mode/2-beamsplitter GBS-style
  interferometer — confirms `beamsplitter_local_matrix` really is
  passive/energy-conserving, not just symplectic in the abstract (a
  symplectic transformation need not conserve photon number in general —
  squeezing is symplectic and clearly does NOT conserve it, since it's
  literally how photons get added to the vacuum — so this is a genuinely
  distinguishing check of the beamsplitter specifically, not redundant
  with the symplectic check).

## Platform notes

Single-threaded numerical code (the homodyne-sampling test's 200k trials
share one RNG sequentially, no threading) — TSan not applicable. Clean
under both `--preset debug` and `--preset asan` (ASan+UBSan).
