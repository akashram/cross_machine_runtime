cmake_minimum_required(VERSION 3.25)

# Phase 12: Machine Learning Library
# Mostly CPU C++ — buildable on Mac. No special hardware required for Phase 12a/b/c.
# Depends on foundation/ (lock-free primitives, work-stealing pool) and cpu_engine/ (SIMD).

# Phase 12a: Classical ML algorithms
add_subdirectory(decision_tree)
add_subdirectory(random_forest)
add_subdirectory(gradient_boosting)
add_subdirectory(svm)
add_subdirectory(knn)
add_subdirectory(kmeans)
add_subdirectory(pca)
add_subdirectory(linear_models)

# Phase 12b: Evaluation framework
add_subdirectory(openml_bench)
add_subdirectory(cross_method)
add_subdirectory(decision_criteria)
add_subdirectory(hyperparam_sensitivity)
add_subdirectory(ensemble)
add_subdirectory(failure_modes)

# Phase 12c: Hyperparameter optimization
add_subdirectory(bayesian_opt)
add_subdirectory(tpe)
add_subdirectory(hyperband)
add_subdirectory(pbt)
