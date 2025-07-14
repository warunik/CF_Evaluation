# Counterfactual Feasibility Analysis Report - HEART_XGBOOST

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 92
- **Feasible Counterfactuals**: 33
- **Feasibility Rate**: 35.9%

## Feature Constraint Analysis
- **Immutable features involved**: 9
- **Partially immutable features involved**: 98
- **Mutable features involved**: 39
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **age**: 0/5 (0.0% feasible)
- **ca**: 29/54 (53.7% feasible)
  - Average change magnitude: 0.768
  - Maximum change magnitude: 1.876
- **chol**: 3/6 (50.0% feasible)
  - Average change magnitude: 36.775
  - Maximum change magnitude: 48.532
- **cp**: 7/14 (50.0% feasible)
  - Average change magnitude: 0.619
  - Maximum change magnitude: 1.140
- **exang**: 4/8 (50.0% feasible)
  - Average change magnitude: 0.558
  - Maximum change magnitude: 0.792
- **fbs**: 1/6 (16.7% feasible)
  - Average change magnitude: 0.529
  - Maximum change magnitude: 0.529
- **oldpeak**: 4/6 (66.7% feasible)
  - Average change magnitude: 0.887
  - Maximum change magnitude: 2.022
- **restecg**: 5/9 (55.6% feasible)
  - Average change magnitude: 0.429
  - Maximum change magnitude: 0.817
- **sex**: 0/4 (0.0% feasible)
- **slope**: 3/10 (30.0% feasible)
  - Average change magnitude: 1.058
  - Maximum change magnitude: 1.864
- **thal**: 8/11 (72.7% feasible)
  - Average change magnitude: 0.651
  - Maximum change magnitude: 0.997
- **thalach**: 3/6 (50.0% feasible)
  - Average change magnitude: 34.635
  - Maximum change magnitude: 47.595
- **trestbps**: 4/7 (57.1% feasible)
  - Average change magnitude: 20.765
  - Maximum change magnitude: 45.385

## Common Feasibility Issues

### age
- age is immutable and cannot be changed (5 instances)

### ca
- ca already satisfies <= 1.646 (current: 0.0) (1 instances)
- ca already satisfies > 0.798 (current: 1.0) (1 instances)
- ca already satisfies > 0.976 (current: 1.0) (1 instances)
- ca already satisfies <= 1.359 (current: 1.0) (1 instances)
- ca already satisfies <= 0.725 (current: 0.0) (1 instances)
- ca already satisfies <= 0.956 (current: 0.0) (1 instances)
- ca already satisfies <= 1.287 (current: 0.0) (1 instances)
- ca already satisfies <= 0.867 (current: 0.0) (1 instances)
- ca already satisfies <= 0.932 (current: 0.0) (1 instances)
- ca already satisfies <= 1.237 (current: 0.0) (1 instances)
- ca already satisfies > 0.691 (current: 3.0) (1 instances)
- ca already satisfies <= 0.54 (current: 0.0) (1 instances)
- ca already satisfies <= 0.753 (current: 0.0) (1 instances)
- ca already satisfies <= 1.022 (current: 0.0) (1 instances)
- ca already satisfies <= 1.005 (current: 1.0) (1 instances)
- ca already satisfies > 0.419 (current: 1.0) (1 instances)
- ca already satisfies <= 0.166 (current: 0.0) (1 instances)
- ca already satisfies <= 1.155 (current: 1.0) (1 instances)
- ca already satisfies <= 1.085 (current: 0.0) (1 instances)
- ca already satisfies <= 0.695 (current: 0.0) (1 instances)
- ca already satisfies <= 0.821 (current: 0.0) (1 instances)
- ca already satisfies > 0.84 (current: 1.0) (1 instances)
- ca already satisfies > 0.942 (current: 1.0) (1 instances)
- ca already satisfies <= 1.191 (current: 0.0) (1 instances)
- ca already satisfies <= 1.255 (current: 1.0) (1 instances)

### chol
- chol already satisfies > 231.823 (current: 244.0) (1 instances)
- chol already satisfies <= 270.64 (current: 242.0) (1 instances)
- chol already satisfies > 217.899 (current: 342.0) (1 instances)

### cp
- cp already satisfies <= 1.413 (current: 1.0) (1 instances)
- cp already satisfies > 0.743 (current: 2.0) (1 instances)
- cp already satisfies <= 0.51 (current: 0.0) (1 instances)
- cp already satisfies <= 0.876 (current: 0.0) (1 instances)
- cp already satisfies > 0.862 (current: 2.0) (1 instances)
- cp already satisfies <= 0.074 (current: 0.0) (1 instances)
- cp already satisfies > 1.654 (current: 2.0) (1 instances)

### exang
- exang already satisfies > 0.148 (current: 1.0) (1 instances)
- exang already satisfies <= 0.326 (current: 0.0) (1 instances)
- exang target value -0.19 outside realistic range [0, 1] (1 instances)
- exang target value -0.484 outside realistic range [0, 1] (1 instances)

### fbs
- fbs target value -0.244 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.206 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.224 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.173 (current: 0.0) (1 instances)
- fbs target value -0.491 outside realistic range [0, 1] (1 instances)

### oldpeak
- oldpeak target value -1.903 outside realistic range [0, 6.2] (1 instances)
- oldpeak target value -1.931 outside realistic range [0, 6.2] (1 instances)

### restecg
- restecg already satisfies <= 0.39 (current: 0.0) (1 instances)
- restecg target value -0.054 outside realistic range [0, 2] (1 instances)
- restecg already satisfies > 0.507 (current: 1.0) (1 instances)
- restecg target value -0.055 outside realistic range [0, 2] (1 instances)

### sex
- sex is immutable and cannot be changed (4 instances)

### slope
- slope target value 2.076 outside realistic range [0, 2] (1 instances)
- slope already satisfies > 1.121 (current: 2.0) (1 instances)
- slope already satisfies > 1.218 (current: 2.0) (1 instances)
- slope already satisfies <= 1.448 (current: 0.0) (1 instances)
- slope already satisfies > 0.634 (current: 2.0) (1 instances)
- slope already satisfies > 1.445 (current: 2.0) (1 instances)
- slope already satisfies > 1.715 (current: 2.0) (1 instances)

### thal
- thal already satisfies > 2.985 (current: 3.0) (1 instances)
- thal target value 3.327 outside realistic range [0, 3] (1 instances)
- thal target value 3.016 outside realistic range [0, 3] (1 instances)

### thalach
- thalach already satisfies > 142.158 (current: 192.0) (1 instances)
- thalach already satisfies > 142.992 (current: 163.0) (1 instances)
- thalach already satisfies > 126.777 (current: 150.0) (1 instances)

### trestbps
- trestbps already satisfies <= 132.339 (current: 117.0) (1 instances)
- trestbps already satisfies <= 121.563 (current: 112.0) (1 instances)
- trestbps already satisfies > 141.176 (current: 160.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 146
- **Feasible conditions**: 71
- **Condition feasibility rate**: 48.6%

