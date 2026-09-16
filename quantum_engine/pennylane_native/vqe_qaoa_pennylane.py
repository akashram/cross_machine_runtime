"""Phase 20 step 7: PennyLane framework-native reimplementation of steps 5
(VQE) and 6 (QAOA), for direct comparison against the hand-rolled
quantum_engine/vqe and quantum_engine/qaoa results.

UNRUN: PennyLane is not installed in this environment (the user
explicitly declined installing it for this session -- see the standing
no-new-local-installs policy, same treatment as JAX before Phase 8's
install and PyTorch/JAX/Ray before Phase 19's). This is real, complete,
correct PennyLane API usage, not a stub -- same convention as
gpu_engine's unrun CUDA kernels and compiler/'s unrun MLIR passes.

The point of this step is PennyLane's real automatic differentiation of
quantum circuits -- qml.qnode + a PennyLane optimizer compute gradients
via the library's own machinery (parameter-shift under the hood for
these gates), not this repo's hand-rolled parameter-shift (vqe.h) or
finite-difference (qaoa.h) code. Both circuits below are built to be
STRUCTURALLY IDENTICAL to their hand-rolled counterparts (same
Hamiltonian, same ansatz shape, same graph instance) so the printed
results are directly comparable once run.

To run once PennyLane is installed:
    pip install pennylane
    python vqe_qaoa_pennylane.py
"""

import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------
# VQE -- same 3-qubit transverse-field-Ising-like Hamiltonian as
# quantum_engine/vqe/vqe_test.cpp's tfim_hamiltonian_terms():
#   H = Z0 + Z1 + Z2 + 0.5*(X0X1 + X1X2)
# Expected comparison point: vqe/README.md's exact ground energy
# -3.124885 (power iteration, cross-checked against a hand-derivable
# closed-form case).

N_QUBITS_VQE = 3
# depth=4 matches vqe/README.md's real finding: depth=1/2 hits a genuine
# expressibility ceiling on this Hamiltonian (plateaus 0.063 above the
# true ground energy across 8 restarts); depth=4 reaches it exactly.
VQE_DEPTH = 4

dev_vqe = qml.device("default.qubit", wires=N_QUBITS_VQE)


def vqe_hamiltonian():
    coeffs = [1.0, 1.0, 1.0, 0.5, 0.5]
    obs = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(2),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliX(1) @ qml.PauliX(2),
    ]
    return qml.Hamiltonian(coeffs, obs)


def vqe_ansatz(theta):
    """Same hardware-efficient ansatz as vqe.h's vqe_ansatz(): `depth`
    layers of (RY on every qubit; CNOT ladder)."""
    idx = 0
    for _ in range(VQE_DEPTH):
        for q in range(N_QUBITS_VQE):
            qml.RY(theta[idx], wires=q)
            idx += 1
        for q in range(N_QUBITS_VQE - 1):
            qml.CNOT(wires=[q, q + 1])


@qml.qnode(dev_vqe)
def vqe_cost(theta):
    vqe_ansatz(theta)
    return qml.expval(vqe_hamiltonian())


def run_vqe(iterations=300, stepsize=0.15, seed=11):
    pnp.random.seed(seed)
    theta = pnp.array(0.1 * pnp.random.randn(N_QUBITS_VQE * VQE_DEPTH), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)

    energies = []
    for _ in range(iterations):
        theta, energy = opt.step_and_cost(vqe_cost, theta)
        energies.append(energy)

    print(f"VQE (PennyLane): start={energies[0]:.6f} end={energies[-1]:.6f}")
    print("  compare against vqe/README.md: exact ground energy -3.124885, hand-rolled depth=4 result -3.124885")
    return energies


# ---------------------------------------------------------------------
# QAOA -- same C5 cycle MaxCut instance as quantum_engine/qaoa/qaoa_test.cpp:
# an odd 5-cycle, classical max cut provably 4 of 5 edges (qaoa/README.md).

N_QUBITS_QAOA = 5
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

dev_qaoa = qml.device("default.qubit", wires=N_QUBITS_QAOA)


def cost_hamiltonian():
    """<C> = sum_edges 0.5*(1 - <Z_i Z_j>) = 0.5*|E| - 0.5*sum<Z_i Z_j>,
    built as one Hamiltonian (identity term carries the constant 0.5*|E|)
    so qml.expval computes the actual expected cut value directly, same
    quantity as qaoa.h's qaoa_expected_cut()."""
    n_edges = len(EDGES)
    coeffs = [0.5 * n_edges] + [-0.5] * n_edges
    obs = [qml.Identity(0)] + [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in EDGES]
    return qml.Hamiltonian(coeffs, obs)


def qaoa_layer(gamma, beta):
    """Same cost/mixer unitaries as qaoa.h: exp(-i*gamma*C) per edge via
    the CNOT-RZ-CNOT ZZ-rotation identity (apply_zz's real-hardware
    equivalent -- PennyLane has no built-in ZZ gate primitive either, so
    this is the same circuit, not a library shortcut around it), then
    RX(2*beta) per qubit for exp(-i*beta*B)."""
    for i, j in EDGES:
        qml.CNOT(wires=[i, j])
        qml.RZ(gamma, wires=j)
        qml.CNOT(wires=[i, j])
    for q in range(N_QUBITS_QAOA):
        qml.RX(2 * beta, wires=q)


@qml.qnode(dev_qaoa)
def qaoa_expected_cut(params, p):
    for q in range(N_QUBITS_QAOA):
        qml.Hadamard(wires=q)
    gammas, betas = params[:p], params[p:]
    for layer in range(p):
        qaoa_layer(gammas[layer], betas[layer])
    return qml.expval(cost_hamiltonian())


def run_qaoa(p, iterations=300, stepsize=0.05, seed=3):
    pnp.random.seed(seed)
    params = pnp.array(0.1 * pnp.random.randn(2 * p) + 0.3, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)

    def negative_cut(pr):
        return -qaoa_expected_cut(pr, p)

    cuts = []
    for _ in range(iterations):
        params, neg_cut = opt.step_and_cost(negative_cut, params)
        cuts.append(-neg_cut)

    print(f"QAOA p={p} (PennyLane): start={cuts[0]:.4f} end={cuts[-1]:.4f}")
    print("  compare against qaoa/README.md: classical max cut 4.0; hand-rolled p=1 -> 3.75 (ratio 0.9375), p=2 -> 4.0 (ratio 1.0)")
    return cuts


if __name__ == "__main__":
    run_vqe()
    for p in (1, 2):
        run_qaoa(p)
