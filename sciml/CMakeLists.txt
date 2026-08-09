cmake_minimum_required(VERSION 3.25)

# Phase 18: Dynamical Systems, SciML & Physics-Informed Architectures.
# Added 2026-08-09, not one of PLAN.md's original 12 phases -- see PLAN.md's
# "Phase 18" section and sciml/DESIGN.md. Fully CPU-portable: no CUDA/Linux/
# MLIR/hardware dependency for any step.

add_subdirectory(ode_solver)
add_subdirectory(stiffness)
