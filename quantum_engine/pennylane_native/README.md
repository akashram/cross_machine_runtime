# pennylane_native

**Status: code-complete, UNRUN — PennyLane is not installed (user declined
this session's install decision).**

## What this is

PLAN.md Phase 20 step 7: a PennyLane framework-native reimplementation of
steps 5 (VQE) and 6 (QAOA), for direct comparison against the hand-rolled
`quantum_engine/vqe` and `quantum_engine/qaoa` results — the same kind of
framework-fluency demonstration Phase 19 built for PyTorch/JAX, specific
to PennyLane (Xanadu's own library) here since it's the most relevant
framework for CV/photonic quantum computing employers.

No CMake target: real Python, not built by the CMake tree, same
non-CMake convention as `gpu_engine/triton_kernels/` and
`framework_native/`.

## Why it's unrun

The standing local-install policy (see project memory — same treatment
JAX got before Phase 8's install, and PyTorch/JAX/Ray before Phase 19's)
requires asking before installing a new toolchain. This session, asked
explicitly, the user declined all four candidate installs for this
phase's work (PennyLane, Strawberry Fields, MPI/OpenMP, MinIO), choosing
to write complete real code now and defer installation. `vqe_qaoa_pennylane.py`
is real, complete, syntactically valid PennyLane API usage (`qml.device`,
`qml.qnode`, `qml.Hamiltonian`, `qml.GradientDescentOptimizer`), not a
stub — same convention as `gpu_engine`'s unrun CUDA kernels and
`compiler/`'s unrun MLIR passes.

`python3 -m py_compile vqe_qaoa_pennylane.py` passes (syntax-checked
locally), but this only confirms the Python is well-formed — it does not
confirm the PennyLane API calls are correct against a specific installed
version, since no PennyLane package is available to import against.

## Design

Both circuits are built to be STRUCTURALLY IDENTICAL to their hand-rolled
counterparts, specifically so the eventual run's output is directly
comparable:

- **VQE**: the same 3-qubit Hamiltonian `H = Z0+Z1+Z2 + 0.5*(X0X1+X1X2)`
  as `vqe/vqe_test.cpp`'s `tfim_hamiltonian_terms()`, the same
  hardware-efficient ansatz shape (`depth=4` layers of RY-then-CNOT-ladder
  — `vqe/README.md`'s real finding that depth=1/2 hits an expressibility
  ceiling on this Hamiltonian is carried over as the depth choice here,
  not re-derived), driven by `qml.GradientDescentOptimizer` instead of
  `vqe.h`'s hand-rolled parameter-shift loop.
- **QAOA**: the same C5 odd-cycle MaxCut instance as `qaoa/qaoa_test.cpp`
  (classical max cut provably 4 of 5 edges), the same cost/mixer circuit
  (PennyLane has no built-in ZZ-rotation gate either, so `qaoa_layer` uses
  the identical `CNOT; RZ; CNOT` identity `apply_zz` does — not a library
  shortcut around the same construction), driven by PennyLane's own
  autodiff through `qml.qnode` rather than `qaoa.h`'s hand-rolled
  finite-difference gradient.
- Both `run_vqe()`/`run_qaoa()` print the hand-rolled comparison numbers
  inline (`vqe/README.md`'s `-3.124885`; `qaoa/README.md`'s `3.75`/`4.0`
  at `p=1`/`p=2`) so running this script directly shows the comparison,
  not just the PennyLane-side number in isolation.

## Results

`TODO: run on [hardware/toolchain]` — needs `pip install pennylane` (ask
before installing, per the standing policy). Once run, fill in:

- VQE: PennyLane's converged energy vs. the hand-rolled `-3.124885`
  (expected to agree closely, since both use the identical Hamiltonian
  and ansatz and both should converge to the same global optimum, modulo
  optimizer/random-seed differences — PennyLane's `default.qubit` device
  is itself an exact state-vector simulator, so no noise should separate
  the two numbers the way it would on real hardware).
- QAOA: PennyLane's `p=1`/`p=2` expected cut values vs. the hand-rolled
  `3.75`/`4.0` (same reasoning — both are exact, noiseless simulations of
  the identical circuit).
- Any real discrepancy found here (rather than the expected close
  agreement) would itself be a genuine finding worth investigating, the
  same way `framework_native/pytorch_transformer`'s C++-vs-PyTorch loss
  comparison was a real cross-implementation check, not a formality.

## Platform notes

Python 3, `pennylane` package required (not installed in `.venv` — see
`framework_native/README.md` for what IS already installed there: torch,
jax, ray, but not pennylane).
