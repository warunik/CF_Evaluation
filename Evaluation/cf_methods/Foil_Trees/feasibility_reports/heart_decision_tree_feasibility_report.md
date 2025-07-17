# Counterfactual Feasibility Analysis Report - HEART_DECISION_TREE

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 103
- **Feasible Counterfactuals**: 32
- **Feasibility Rate**: 31.1%

## Feature Constraint Analysis
- **Immutable features involved**: 17
- **Partially immutable features involved**: 88
- **Mutable features involved**: 65
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **age**: 0/12 (0.0% feasible)
- **ca**: 12/23 (52.2% feasible)
  - Average change magnitude: 0.737
  - Maximum change magnitude: 1.860
- **chol**: 7/14 (50.0% feasible)
  - Average change magnitude: 57.924
  - Maximum change magnitude: 93.082
- **cp**: 15/20 (75.0% feasible)
  - Average change magnitude: 0.995
  - Maximum change magnitude: 2.527
- **exang**: 3/7 (42.9% feasible)
  - Average change magnitude: 0.343
  - Maximum change magnitude: 0.699
- **fbs**: 1/9 (11.1% feasible)
  - Average change magnitude: 0.616
  - Maximum change magnitude: 0.616
- **oldpeak**: 3/9 (33.3% feasible)
  - Average change magnitude: 1.020
  - Maximum change magnitude: 1.511
- **restecg**: 2/8 (25.0% feasible)
  - Average change magnitude: 0.700
  - Maximum change magnitude: 0.778
- **sex**: 0/5 (0.0% feasible)
- **slope**: 2/8 (25.0% feasible)
  - Average change magnitude: 0.988
  - Maximum change magnitude: 1.446
- **thal**: 19/29 (65.5% feasible)
  - Average change magnitude: 0.525
  - Maximum change magnitude: 1.623
- **thalach**: 4/12 (33.3% feasible)
  - Average change magnitude: 16.288
  - Maximum change magnitude: 35.060
- **trestbps**: 3/14 (21.4% feasible)
  - Average change magnitude: 7.598
  - Maximum change magnitude: 19.770

## Common Feasibility Issues

### age
- age is immutable and cannot be changed (12 instances)

### ca
- ca already satisfies <= 1.023 (current: 0.0) (1 instances)
- ca already satisfies <= 0.775 (current: 0.0) (1 instances)
- ca already satisfies <= 1.66 (current: 0.0) (1 instances)
- ca already satisfies > 1.119 (current: 2.0) (1 instances)
- ca already satisfies <= 0.859 (current: 0.0) (1 instances)
- ca already satisfies > 0.277 (current: 1.0) (1 instances)
- ca already satisfies <= 1.461 (current: 0.0) (1 instances)
- ca already satisfies <= 0.846 (current: 0.0) (1 instances)
- ca already satisfies > 0.23 (current: 3.0) (1 instances)
- ca already satisfies > 0.845 (current: 1.0) (1 instances)
- ca already satisfies <= 0.421 (current: 0.0) (1 instances)

### chol
- chol already satisfies <= 236.337 (current: 233.0) (1 instances)
- chol already satisfies > 231.986 (current: 308.0) (1 instances)
- chol already satisfies > 234.905 (current: 309.0) (1 instances)
- chol already satisfies > 203.763 (current: 215.0) (1 instances)
- chol already satisfies > 223.307 (current: 230.0) (1 instances)
- chol already satisfies > 177.552 (current: 201.0) (1 instances)
- chol already satisfies > 203.932 (current: 282.0) (1 instances)

### cp
- cp target value -0.044 outside realistic range [0, 3] (1 instances)
- cp already satisfies > 1.321 (current: 2.0) (1 instances)
- cp already satisfies > 0.5 (current: 2.0) (1 instances)
- cp already satisfies <= 0.39 (current: 0.0) (1 instances)
- cp target value -0.514 outside realistic range [0, 3] (1 instances)

### exang
- exang already satisfies <= 0.266 (current: 0.0) (1 instances)
- exang target value -0.158 outside realistic range [0, 1] (1 instances)
- exang target value -0.288 outside realistic range [0, 1] (1 instances)
- exang target value -0.054 outside realistic range [0, 1] (1 instances)

### fbs
- fbs already satisfies <= 0.064 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.433 (current: 0.0) (1 instances)
- fbs target value -0.282 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.206 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.324 (current: 0.0) (1 instances)
- fbs target value -0.465 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.379 (current: 0.0) (1 instances)
- fbs already satisfies <= 0.051 (current: 0.0) (1 instances)

### oldpeak
- oldpeak already satisfies > 0.713 (current: 1.9) (1 instances)
- oldpeak already satisfies > 1.218 (current: 1.4) (1 instances)
- oldpeak already satisfies <= 2.031 (current: 0.0) (1 instances)
- oldpeak already satisfies <= 2.267 (current: 0.0) (1 instances)
- oldpeak target value -0.225 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies <= 1.294 (current: 0.0) (1 instances)

### restecg
- restecg already satisfies > 0.559 (current: 1.0) (1 instances)
- restecg already satisfies > 0.653 (current: 1.0) (1 instances)
- restecg target value -0.497 outside realistic range [0, 2] (1 instances)
- restecg already satisfies > 0.48 (current: 1.0) (1 instances)
- restecg target value -0.031 outside realistic range [0, 2] (1 instances)
- restecg already satisfies <= 1.09 (current: 0.0) (1 instances)

### sex
- sex is immutable and cannot be changed (5 instances)

### slope
- slope already satisfies <= 1.637 (current: 1.0) (1 instances)
- slope target value 2.6 outside realistic range [0, 2] (1 instances)
- slope target value 2.28 outside realistic range [0, 2] (1 instances)
- slope already satisfies <= 1.088 (current: 1.0) (1 instances)
- slope already satisfies > 0.961 (current: 1.0) (1 instances)
- slope already satisfies <= 1.265 (current: 0.0) (1 instances)

### thal
- thal already satisfies > 2.508 (current: 3.0) (1 instances)
- thal already satisfies <= 2.659 (current: 2.0) (1 instances)
- thal already satisfies <= 2.436 (current: 2.0) (1 instances)
- thal already satisfies > 1.894 (current: 2.0) (1 instances)
- thal already satisfies <= 2.246 (current: 2.0) (1 instances)
- thal already satisfies > 1.719 (current: 2.0) (1 instances)
- thal already satisfies > 2.604 (current: 3.0) (1 instances)
- thal already satisfies > 2.374 (current: 3.0) (1 instances)
- thal already satisfies <= 2.001 (current: 2.0) (1 instances)
- thal already satisfies <= 2.291 (current: 2.0) (1 instances)

### thalach
- thalach already satisfies > 99.506 (current: 152.0) (1 instances)
- thalach already satisfies <= 154.056 (current: 152.0) (1 instances)
- thalach already satisfies <= 145.335 (current: 71.0) (1 instances)
- thalach already satisfies > 141.154 (current: 188.0) (1 instances)
- thalach already satisfies > 114.495 (current: 170.0) (1 instances)
- thalach already satisfies <= 152.999 (current: 130.0) (1 instances)
- thalach already satisfies > 129.404 (current: 144.0) (1 instances)
- thalach already satisfies > 138.153 (current: 147.0) (1 instances)

### trestbps
- trestbps already satisfies > 111.877 (current: 148.0) (1 instances)
- trestbps already satisfies > 135.367 (current: 138.0) (1 instances)
- trestbps already satisfies <= 134.424 (current: 130.0) (1 instances)
- trestbps already satisfies > 113.979 (current: 125.0) (1 instances)
- trestbps already satisfies > 125.787 (current: 132.0) (1 instances)
- trestbps already satisfies <= 131.259 (current: 125.0) (1 instances)
- trestbps already satisfies > 116.871 (current: 128.0) (1 instances)
- trestbps already satisfies > 119.635 (current: 123.0) (1 instances)
- trestbps already satisfies > 139.873 (current: 160.0) (1 instances)
- trestbps already satisfies <= 150.972 (current: 128.0) (1 instances)
- trestbps already satisfies > 132.685 (current: 146.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 170
- **Feasible conditions**: 71
- **Condition feasibility rate**: 41.8%

