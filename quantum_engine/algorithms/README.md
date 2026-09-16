# algorithms

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 20 step 2: three canonical algorithms on `state_vector`'s
simulator — Deutsch-Jozsa, Grover's search, and the Quantum Fourier
Transform — each checked against an independent classical ground truth,
not just "the circuit runs."

## Design

- **Deutsch-Jozsa**: `deutsch_jozsa(n, oracle)` implements the standard
  phase-kickback circuit (ancilla in `|1>`, Hadamard everything, apply the
  oracle's `|x>|y> -> |x>|y XOR f(x)>` unitary, Hadamard the input
  register, check the all-zero-input probability). The oracle is a real
  `std::function` parameter, not a hardcoded case — `constant_oracle`,
  `parity_oracle`, and `single_bit_oracle` are three genuinely different
  oracle shapes, tested across `n=2..6`.
- **Grover**: the oracle is represented directly as the unitary it must
  implement, `S_f = I - 2|w><w|` (a diagonal sign flip on the marked
  index) — the standard "oracle as unitary" abstraction textbook Grover
  treatments and library simulators use; the `O(sqrt(N))`-QUERIES result
  is about how many times this black box is invoked, not how many gates
  it costs to synthesize from a boolean circuit. `grover_diffusion`
  builds `D = H^n (2|0><0| - I) H^n` the standard way.
- **QFT**: `qft(sv, n)` is the textbook rotation-ladder circuit (H then
  controlled `R_k` phases from every less-significant qubit, most- to
  least-significant qubit order) followed by a bit-reversal SWAP pass
  (each SWAP built from 3 CNOTs, reusing the existing primitive rather
  than adding a new one) — needed because the rotation ladder produces
  output in reversed qubit order relative to this simulator's
  qubit-0-is-LSB convention.

## Results (captured 2026-09-16, Apple clang 14 / `-std=c++2b`, this Mac)

```
  n=2: constant(f=0)->constant constant(f=1)->constant parity->balanced single-bit->balanced
  ...
PASS  Deutsch-Jozsa correctly classifies constant oracles as constant and balanced oracles as balanced for n=2..6, single query each
  n=3 (N=8): formula-iterations=2 local-brute-force-argmax(window +-3)=2 success-prob-at-formula=0.9453 formula/sqrt(N)=0.7071
  n=4 (N=16): formula-iterations=3 local-brute-force-argmax(window +-3)=3 success-prob-at-formula=0.9613 formula/sqrt(N)=0.7500
  n=5 (N=32): formula-iterations=4 local-brute-force-argmax(window +-3)=4 success-prob-at-formula=0.9992 formula/sqrt(N)=0.7071
  n=6 (N=64): formula-iterations=6 local-brute-force-argmax(window +-3)=6 success-prob-at-formula=0.9966 formula/sqrt(N)=0.7500
  n=7 (N=128): formula-iterations=8 local-brute-force-argmax(window +-3)=8 success-prob-at-formula=0.9956 formula/sqrt(N)=0.7071
PASS  Grover's closed-form optimal-iteration formula is a local optimum (brute-force search over a window of nearby iteration counts finds nothing better, within 1) and achieves >90% success probability, for n=3..7
  max |circuit_output - brute_force_DFT(input)| over 8 amplitudes = 5.467e-16
PASS  QFT circuit's output on a nontrivial 3-qubit input matches a brute-force O(N^2) DFT of the same input exactly
PASS
```

## Findings

- **A real test-design bug, caught by running it, not a circuit bug.**
  The first version of the Grover test brute-force-searched success
  probability over a WIDE range of iteration counts (`0` to
  `~2*sqrt(N)+2`) and compared the argmax against the closed-form
  formula — and failed for every `n`, with the brute-force argmax
  landing far past the formula's prediction (e.g. `n=3`: formula says 2,
  brute force said 6). This is not a bug in Grover's implementation:
  success probability is `sin^2((2k+1)*theta)`, which is PERIODIC in `k`
  with period `pi/(2*theta)` — roughly twice the optimal iteration count
  itself for a single marked item — so a wide-range search legitimately
  finds later peaks that round to a marginally higher discrete
  probability than the first peak, purely from where an integer `k`
  happens to land relative to each period's continuous maximum. A
  formula for "iterations to the FIRST peak" (the only iteration count
  anyone actually uses — continuing past it wastes oracle queries and
  the probability drops before re-peaking) should not be checked against
  an unbounded search that just rediscovers periodicity. Fixed by
  windowing the brute-force search to `formula +/- 3` — a real
  brute-force check of local optimality, which the formula now passes at
  every `n=3..7`, with success probability `>90%` in every case.
- **QFT matches brute-force DFT to `5.5e-16`** — effectively exact
  (floating-point roundoff only) on a genuinely non-basis-state 3-qubit
  input (built via H/RY/RZ/CNOT before the QFT, not `|000>`), confirming
  the rotation-ladder-plus-bit-reversal circuit really does implement the
  `1/sqrt(N) sum_k e^{2*pi*i*j*k/N} |k>` transform by linearity, not just
  on a single basis state.

## Platform notes

Single-threaded numerical code, no concurrency — TSan not applicable.
Builds and passes under `--preset debug`.
