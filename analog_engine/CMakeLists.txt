cmake_minimum_required(VERSION 3.25)

# Phase 17: Analog & Unconventional Compute Hardware. Added 2026-08-09, not
# one of PLAN.md's original 12 phases -- see PLAN.md's "Phase 17" section
# and analog_engine/DESIGN.md. Fully CPU-portable: unlike Phase 3/7/8/15,
# there is no toolchain to gate behind at all (no analog/neuromorphic
# silicon exists to rent), so every step is real, locally-runnable
# numeric/simulation code.

add_subdirectory(device_model)
add_subdirectory(crossbar_mac)
add_subdirectory(nvm_comparison)
add_subdirectory(energy_model)
add_subdirectory(dataflow_model)
add_subdirectory(systolic_sweep)
