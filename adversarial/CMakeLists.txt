# Phase 14: Adversarial Robustness. Added 2026-07-28, not one of PLAN.md's
# original 12 phases — see PLAN.md's "Phase 14" section and
# adversarial/DESIGN.md. Fully CPU-portable: builds on
# distributed_training/autograd (the generic reverse-mode tape) and
# distributed_training/full_training_loop's MLP model, both already
# code-complete. No CUDA/Linux/MLIR dependency for any step.

add_subdirectory(task)
add_subdirectory(input_gradients)
add_subdirectory(fgsm)
add_subdirectory(pgd)
add_subdirectory(vulnerability_measurement)
add_subdirectory(adversarial_training)
add_subdirectory(robustness_tradeoff)
add_subdirectory(transferability)
add_subdirectory(randomized_smoothing)
