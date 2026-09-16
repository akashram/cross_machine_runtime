# cost_model (qpu_landscape)

**Status: code-complete AND locally run — pure CPU, no external dependency.
Literature/vendor-doc-grounded data, honestly labeled as illustrative.**

## What this measures

PLAN.md Phase 20 step 10: a written comparison of IBM Quantum, AWS Braket
(aggregating IonQ, Rigetti, QuEra), Azure Quantum (aggregating IonQ,
Quantinuum, Rigetti), and Xanadu Cloud — current published qubit counts,
gate error rates, coherence times, and access/cost model per provider,
feeding into the eventual hardware-validation choice. Same honest-
labeling convention as `analog_engine/nvm_comparison`.

## Disclosed limitation (stated up front, same as the header comment)

The figures in `qpu_landscape.h` are **illustrative, order-of-magnitude
representative values** drawn from each vendor's general public
technology documentation and widely-reported specifications — **not
live-scraped or independently verified against a specific dated source**
(no network access was used to build this table). Real published qubit
counts, gate error rates, and pricing change frequently, and vary by
device/generation within a single provider's own fleet. Treat every
number here as "the right order of magnitude and qualitative ranking,"
not a precise citation — the same caveat `nvm_comparison.h` gives its own
numbers, applied to a domain (cloud QPU access) that changes even faster
than memory-device fabrication figures do.

## Design

- One `QpuDevice` entry per (cloud provider, hardware vendor) pair —
  `AWS Braket`/`IonQ` and `Azure Quantum`/`IonQ`-class-but-actually-
  Quantinuum are DIFFERENT rows, since the cloud access point and the
  actual chip maker are genuinely different axes (Rigetti hardware is
  reachable through BOTH AWS Braket and Azure Quantum, for instance —
  represented as two separate rows with identical hardware figures,
  not collapsed into one, since the access/pricing model differs).
- `QpuModality` distinguishes superconducting, trapped-ion, neutral-atom,
  and photonic (CV) — genuinely different physical implementations with
  different native metrics, not points on one scale.
- Xanadu's entry reports **squeezed MODES, not qubits** (`216`, in the
  same field as other rows' qubit counts, for table-layout convenience
  only) — `approx_two_qubit_gate_error_pct=0.0` and
  `approx_coherence_time_us=0.0` are sentinel "not comparable" values,
  and `mean_gate_error_for_modality`/`mean_coherence_for_modality`
  explicitly SKIP any entry with a non-positive value, so Xanadu's row
  never silently pollutes a cross-modality average with a meaningless
  zero.

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  provider        vendor       modality           qubits    2Q-err% coherence-us
  IBM Quantum     IBM          superconducting       130       0.50        150.0
  AWS Braket      IonQ         trapped-ion            32       0.40    1000000.0
  AWS Braket      Rigetti      superconducting        80       2.00         20.0
  AWS Braket      QuEra        neutral-atom          256       1.50          1.0
  Azure Quantum   Quantinuum   trapped-ion            56       0.10    1000000.0
  Azure Quantum   Rigetti      superconducting        80       2.00         20.0
  Xanadu Cloud    Xanadu       photonic (CV)         216       0.00          0.0
PASS  landscape table covers at least 4 distinct hardware entries across the 4 named providers

  mean 2Q gate error: trapped-ion=0.250% superconducting=1.500%
PASS  trapped-ion devices have a LOWER mean two-qubit gate error than superconducting devices -- the real, textbook higher-fidelity/slower-gates tradeoff
  mean coherence time: trapped-ion=1000000.0us superconducting=63.3us
PASS  trapped-ion devices have a LONGER mean coherence time than superconducting devices -- the other half of the same tradeoff
PASS  Xanadu's photonic entry is correctly excluded from the gate-error comparison (reports squeezed modes, not a directly comparable two-qubit-gate-error figure) rather than silently averaged in
PASS
```

## Discussion: which provider fits which of this phase's steps

Deliberately NOT collapsed into a single composite "winner" score (unlike
`nvm_comparison`'s figure-of-merit ranking) — these are genuinely
different architectures solving different problems, and a single number
would misrepresent that the way it wouldn't for four NVM technologies all
targeting the same crossbar-cell role:

- **Steps 1-6 (gate-based: state vector, algorithms, noise, QEC, VQE,
  QAOA)** map most directly onto the gate-based providers — **IBM
  Quantum** for its broad free-tier access (lowest barrier to a first
  real-hardware run), or **IonQ**/**Quantinuum** (via Braket/Azure) if
  the priority is matching this phase's exact-simulation results as
  closely as real noisy hardware can (their much lower two-qubit gate
  error, per the table above, means less noise-induced deviation from
  the noiseless results already measured in steps 1-6).
- **Step 7 (PennyLane)** works against any PennyLane-supported backend —
  no provider preference from the framework choice itself, though
  Xanadu's own hardware (below) is the one PennyLane was originally built
  around.
- **Steps 8-9 (CV/photonic, QPU backend registration)** point most
  directly at **Xanadu Cloud** — the SAME software stack (PennyLane) this
  phase already writes real, complete client code for in step 7, reaching
  the SAME formalism (Gaussian states, squeezing, GBS) this phase already
  hand-rolled in step 8, on real photonic hardware rather than a
  classical simulation of it. This is the most load-bearing entry in the
  whole table for THIS repo's specific next step, not a generic "best
  QPU" ranking.
- **QuEra (neutral-atom, via Braket)** doesn't map onto anything built in
  this phase at all — it targets analog Hamiltonian simulation, a
  genuinely different programming model from every gate-based circuit
  steps 1-7 build. Included in the table for completeness (PLAN.md names
  QuEra explicitly as part of the AWS Braket aggregation) but not a
  candidate for this phase's own hardware-validation pass.

## Platform notes

Single-threaded, pure data/comparison code — TSan not applicable. Clean
under both `--preset debug` and `--preset asan` (ASan+UBSan).
