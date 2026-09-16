cmake_minimum_required(VERSION 3.25)

# Phase 20: Quantum Computing & Hybrid Quantum-Classical Compute. Added
# 2026-09-16, not one of PLAN.md's original 12 phases -- see PLAN.md's
# "Phase 20" section and quantum_engine/DESIGN.md. Unlike Phase 17 (no
# rentable silicon exists at all) or Phase 3/7/8/15 (real hardware exists
# but is toolchain/rental-gated), small-qubit-count classical simulation is
# the exact, correct physics -- not a stand-in -- so almost every step here
# is real, locally-runnable code. Steps 7 (PennyLane) and the QPU-hardware
# half of step 9 are the only genuinely toolchain/hardware-gated pieces;
# see each step's own README.

add_subdirectory(state_vector)
add_subdirectory(algorithms)
add_subdirectory(noise_model)
add_subdirectory(qec)
add_subdirectory(vqe)

# step 7 (pennylane_native/) has no CMake target: real PennyLane code,
# unrun, same non-CMake convention as gpu_engine/triton_kernels and
# framework_native/ -- see quantum_engine/pennylane_native/README.md.
