# Counterfactual Feasibility Analysis - Summary Report

## Overview
- **Datasets analyzed**: 5
- **Dataset-model combinations**: 15

## Dataset-wise Results

### HEART

| Model | Total Analyzed | Feasible | Feasibility Rate |
|-------|---------------|----------|------------------|
| mlp | 119 | 26 | 21.8% |
| decision_tree | 103 | 32 | 31.1% |
| logistic_regression | 108 | 16 | 14.8% |
| random_forest | 105 | 36 | 34.3% |
| xgboost | 92 | 33 | 35.9% |
| **Total** | **527** | **143** | **27.1%** |

### DIABETES

| Model | Total Analyzed | Feasible | Feasibility Rate |
|-------|---------------|----------|------------------|
| mlp | 93 | 21 | 22.6% |
| decision_tree | 73 | 17 | 23.3% |
| logistic_regression | 99 | 21 | 21.2% |
| random_forest | 83 | 14 | 16.9% |
| xgboost | 90 | 14 | 15.6% |
| **Total** | **438** | **87** | **19.9%** |

### BANK

| Model | Total Analyzed | Feasible | Feasibility Rate |
|-------|---------------|----------|------------------|
| mlp | 4365 | 165 | 3.8% |
| decision_tree | 4331 | 253 | 5.8% |
| logistic_regression | 543 | 20 | 3.7% |
| random_forest | 2429 | 186 | 7.7% |
| xgboost | 2259 | 254 | 11.2% |
| **Total** | **13927** | **878** | **6.3%** |

## Model-wise Results

| Model | Total Analyzed | Feasible | Feasibility Rate |
|-------|---------------|----------|------------------|
| mlp | 4577 | 212 | 4.6% |
| decision_tree | 4507 | 302 | 6.7% |
| logistic_regression | 750 | 57 | 7.6% |
| random_forest | 2617 | 236 | 9.0% |
| xgboost | 2441 | 301 | 12.3% |
