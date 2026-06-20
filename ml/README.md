# Phase 12: Machine Learning Library

**Status: STUB — CPU C++, buildable on any platform including Mac.**

## Overview
Classical ML algorithms (CART, RF, GBT, SVM, k-NN, k-means, PCA) using
SIMD from cpu_engine/ and lock-free parallelism from foundation/.
Benchmarked against LightGBM/sklearn on OpenML CC-18.

## Phase 12a: Algorithms

| Directory | Algorithm | Key feature |
|-----------|-----------|-------------|
| decision_tree | CART | SIMD split search, Gini/entropy |
| random_forest | Random Forest | Work-stealing parallelism, OOB error |
| gradient_boosting | GBT | Newton steps, histogram-based splits |
| svm | SVM (SMO) | RBF/poly/linear kernels, kernel caching |
| knn | k-NN | KD-tree + SIMD distance |
| kmeans | k-means++ | Lloyd's + SIMD centroid update |
| pca | PCA | Randomized SVD (Halko et al.) |
| linear_models | SGD + L-BFGS | Elastic net, logistic regression |

## Phase 12b: Evaluation

| Directory | What |
|-----------|------|
| openml_bench | Run all 18 OpenML CC-18 datasets |
| cross_method | Compare all algorithms per dataset |
| decision_criteria | When to use what (decision guide) |
| hyperparam_sensitivity | Sweep key hyperparameters per algo |
| ensemble | Stacking, blending, diversity analysis |
| failure_modes | One concrete failure per algorithm |

## Phase 12c: Hyperparameter Optimization

| Directory | What |
|-----------|------|
| bayesian_opt | GP surrogate + Expected Improvement |
| tpe | Tree-structured Parzen Estimator |
| hyperband | Successive halving + early stopping |
| pbt | Population-based training |

## Hardware notes
- Required: any machine (Mac, Linux) — no GPU/cloud needed
- Performance targets: GBT matches LightGBM accuracy, beats it throughput at >1M rows
