# Counterfactual Feasibility Analysis Report - HEART_LOGISTIC_REGRESSION

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 108
- **Feasible Counterfactuals**: 16
- **Feasibility Rate**: 14.8%

## Feature Constraint Analysis
- **Immutable features involved**: 32
- **Partially immutable features involved**: 86
- **Mutable features involved**: 84
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **age**: 0/5 (0.0% feasible)
- **ca**: 10/19 (52.6% feasible)
  - Average change magnitude: 1.167
  - Maximum change magnitude: 2.669
- **chol**: 2/5 (40.0% feasible)
  - Average change magnitude: 61.284
  - Maximum change magnitude: 120.443
- **cp**: 13/29 (44.8% feasible)
  - Average change magnitude: 0.972
  - Maximum change magnitude: 1.824
- **exang**: 5/14 (35.7% feasible)
  - Average change magnitude: 0.647
  - Maximum change magnitude: 0.813
- **fbs**: 3/12 (25.0% feasible)
  - Average change magnitude: 0.768
  - Maximum change magnitude: 0.982
- **oldpeak**: 8/18 (44.4% feasible)
  - Average change magnitude: 1.151
  - Maximum change magnitude: 2.806
- **restecg**: 5/8 (62.5% feasible)
  - Average change magnitude: 0.585
  - Maximum change magnitude: 0.670
- **sex**: 0/27 (0.0% feasible)
- **slope**: 5/11 (45.5% feasible)
  - Average change magnitude: 0.512
  - Maximum change magnitude: 0.977
- **thal**: 9/19 (47.4% feasible)
  - Average change magnitude: 0.507
  - Maximum change magnitude: 1.652
- **thalach**: 7/18 (38.9% feasible)
  - Average change magnitude: 28.937
  - Maximum change magnitude: 82.347
- **trestbps**: 5/17 (29.4% feasible)
  - Average change magnitude: 11.231
  - Maximum change magnitude: 21.995

## Common Feasibility Issues

### age
- age is immutable and cannot be changed (5 instances)

### ca
- ca already satisfies > 1.177 (current: 2.0) (1 instances)
- ca target value -0.428 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 0.8 (current: 0.0) (1 instances)
- ca already satisfies <= 0.846 (current: 0.0) (1 instances)
- ca already satisfies <= 0.713 (current: 0.0) (1 instances)
- ca already satisfies > 0.596 (current: 2.0) (1 instances)
- ca already satisfies <= 1.3 (current: 0.0) (1 instances)
- ca already satisfies <= 1.465 (current: 0.0) (1 instances)
- ca already satisfies <= 0.17 (current: 0.0) (1 instances)

### chol
- chol already satisfies > 224.092 (current: 275.0) (1 instances)
- chol already satisfies > 230.802 (current: 341.0) (1 instances)
- chol already satisfies <= 231.407 (current: 177.0) (1 instances)

### cp
- cp already satisfies > 1.458 (current: 2.0) (1 instances)
- cp already satisfies <= 1.302 (current: 1.0) (1 instances)
- cp already satisfies <= 1.111 (current: 1.0) (1 instances)
- cp already satisfies > 0.425 (current: 2.0) (1 instances)
- cp target value -0.279 outside realistic range [0, 3] (1 instances)
- cp already satisfies > 0.459 (current: 2.0) (1 instances)
- cp already satisfies > 0.335 (current: 1.0) (1 instances)
- cp already satisfies > 1.284 (current: 2.0) (1 instances)
- cp target value -0.353 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 2.448 (current: 1.0) (1 instances)
- cp already satisfies <= 1.346 (current: 1.0) (1 instances)
- cp target value -0.336 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 1.799 (current: 0.0) (1 instances)
- cp already satisfies > 0.443 (current: 1.0) (1 instances)
- cp already satisfies <= 0.18 (current: 0.0) (1 instances)
- cp already satisfies <= 0.845 (current: 0.0) (1 instances)

### exang
- exang target value -0.378 outside realistic range [0, 1] (1 instances)
- exang already satisfies <= 0.488 (current: 0.0) (1 instances)
- exang target value -0.044 outside realistic range [0, 1] (1 instances)
- exang target value -0.294 outside realistic range [0, 1] (1 instances)
- exang already satisfies <= 0.348 (current: 0.0) (1 instances)
- exang already satisfies <= 0.637 (current: 0.0) (1 instances)
- exang target value -0.161 outside realistic range [0, 1] (1 instances)
- exang already satisfies <= 0.447 (current: 0.0) (1 instances)
- exang already satisfies <= 0.346 (current: 0.0) (1 instances)

### fbs
- fbs target value -0.358 outside realistic range [0, 1] (1 instances)
- fbs target value -0.041 outside realistic range [0, 1] (1 instances)
- fbs target value -0.212 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.173 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.055 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.063 (current: 0.0) (1 instances)
- fbs target value -0.236 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.139 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.186 (current: 0.0) (1 instances)

### oldpeak
- oldpeak already satisfies <= 0.93 (current: 0.9) (1 instances)
- oldpeak already satisfies <= 1.453 (current: 0.0) (1 instances)
- oldpeak target value -0.754 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies <= 0.609 (current: 0.0) (1 instances)
- oldpeak already satisfies <= 0.835 (current: 0.0) (1 instances)
- oldpeak already satisfies <= 0.678 (current: 0.0) (1 instances)
- oldpeak already satisfies <= 2.198 (current: 0.8) (1 instances)
- oldpeak target value -0.326 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies > 0.802 (current: 1.9) (1 instances)
- oldpeak already satisfies > 0.682 (current: 0.7) (1 instances)

### restecg
- restecg already satisfies > 0.387 (current: 1.0) (1 instances)
- restecg already satisfies > 0.525 (current: 1.0) (1 instances)
- restecg already satisfies > 0.723 (current: 2.0) (1 instances)

### sex
- sex is immutable and cannot be changed (27 instances)

### slope
- slope already satisfies <= 1.124 (current: 1.0) (1 instances)
- slope already satisfies > 0.998 (current: 1.0) (1 instances)
- slope already satisfies <= 1.432 (current: 1.0) (1 instances)
- slope already satisfies > 0.408 (current: 1.0) (1 instances)
- slope already satisfies <= 1.279 (current: 1.0) (1 instances)
- slope already satisfies > 0.849 (current: 2.0) (1 instances)

### thal
- thal already satisfies <= 2.74 (current: 2.0) (1 instances)
- thal already satisfies > 1.982 (current: 3.0) (1 instances)
- thal already satisfies > 2.516 (current: 3.0) (1 instances)
- thal already satisfies <= 2.107 (current: 2.0) (1 instances)
- thal already satisfies <= 2.562 (current: 2.0) (1 instances)
- thal already satisfies <= 2.649 (current: 2.0) (1 instances)
- thal already satisfies <= 2.118 (current: 2.0) (1 instances)
- thal target value 3.226 outside realistic range [0, 3] (1 instances)
- thal already satisfies <= 2.318 (current: 2.0) (1 instances)
- thal already satisfies > 1.561 (current: 2.0) (1 instances)

### thalach
- thalach already satisfies > 149.277 (current: 160.0) (1 instances)
- thalach already satisfies <= 153.006 (current: 109.0) (1 instances)
- thalach already satisfies <= 168.964 (current: 165.0) (1 instances)
- thalach already satisfies > 113.968 (current: 141.0) (1 instances)
- thalach already satisfies > 103.903 (current: 164.0) (1 instances)
- thalach already satisfies > 128.097 (current: 151.0) (1 instances)
- thalach already satisfies > 141.408 (current: 173.0) (1 instances)
- thalach already satisfies > 136.025 (current: 143.0) (1 instances)
- thalach already satisfies > 108.999 (current: 157.0) (1 instances)
- thalach already satisfies > 145.286 (current: 173.0) (1 instances)
- thalach already satisfies <= 147.466 (current: 133.0) (1 instances)

### trestbps
- trestbps already satisfies <= 125.464 (current: 117.0) (1 instances)
- trestbps already satisfies > 111.785 (current: 120.0) (1 instances)
- trestbps already satisfies <= 148.918 (current: 138.0) (1 instances)
- trestbps already satisfies > 114.948 (current: 138.0) (1 instances)
- trestbps already satisfies <= 145.756 (current: 140.0) (1 instances)
- trestbps already satisfies <= 128.477 (current: 118.0) (1 instances)
- trestbps already satisfies > 130.079 (current: 152.0) (1 instances)
- trestbps already satisfies <= 159.328 (current: 140.0) (1 instances)
- trestbps already satisfies > 111.433 (current: 132.0) (1 instances)
- trestbps already satisfies > 132.155 (current: 134.0) (1 instances)
- trestbps already satisfies > 118.582 (current: 130.0) (1 instances)
- trestbps already satisfies <= 138.525 (current: 110.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 202
- **Feasible conditions**: 72
- **Condition feasibility rate**: 35.6%

