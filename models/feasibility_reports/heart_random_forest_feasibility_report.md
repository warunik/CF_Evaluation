# Counterfactual Feasibility Analysis Report - HEART_RANDOM_FOREST

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 105
- **Feasible Counterfactuals**: 36
- **Feasibility Rate**: 34.3%

## Feature Constraint Analysis
- **Immutable features involved**: 24
- **Partially immutable features involved**: 115
- **Mutable features involved**: 56
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **age**: 0/13 (0.0% feasible)
- **ca**: 16/30 (53.3% feasible)
  - Average change magnitude: 0.921
  - Maximum change magnitude: 2.580
- **chol**: 4/8 (50.0% feasible)
  - Average change magnitude: 50.364
  - Maximum change magnitude: 80.001
- **cp**: 16/31 (51.6% feasible)
  - Average change magnitude: 1.234
  - Maximum change magnitude: 2.564
- **exang**: 6/12 (50.0% feasible)
  - Average change magnitude: 0.466
  - Maximum change magnitude: 0.725
- **fbs**: 1/6 (16.7% feasible)
  - Average change magnitude: 0.958
  - Maximum change magnitude: 0.958
- **oldpeak**: 6/9 (66.7% feasible)
  - Average change magnitude: 2.185
  - Maximum change magnitude: 3.902
- **restecg**: 5/12 (41.7% feasible)
  - Average change magnitude: 0.597
  - Maximum change magnitude: 0.842
- **sex**: 0/11 (0.0% feasible)
- **slope**: 11/20 (55.0% feasible)
  - Average change magnitude: 0.784
  - Maximum change magnitude: 1.691
- **thal**: 13/22 (59.1% feasible)
  - Average change magnitude: 0.614
  - Maximum change magnitude: 0.910
- **thalach**: 8/14 (57.1% feasible)
  - Average change magnitude: 18.906
  - Maximum change magnitude: 42.537
- **trestbps**: 5/7 (71.4% feasible)
  - Average change magnitude: 16.384
  - Maximum change magnitude: 37.189

## Common Feasibility Issues

### age
- age is immutable and cannot be changed (13 instances)

### ca
- ca already satisfies <= 0.426 (current: 0.0) (1 instances)
- ca target value -0.65 outside realistic range [0, 4] (1 instances)
- ca target value -0.725 outside realistic range [0, 4] (1 instances)
- ca target value -0.511 outside realistic range [0, 4] (1 instances)
- ca target value -0.182 outside realistic range [0, 4] (1 instances)
- ca target value -0.263 outside realistic range [0, 4] (1 instances)
- ca already satisfies > 0.358 (current: 1.0) (1 instances)
- ca target value -0.262 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 0.465 (current: 0.0) (1 instances)
- ca already satisfies > 0.833 (current: 1.0) (1 instances)
- ca already satisfies <= 0.67 (current: 0.0) (1 instances)
- ca target value -0.234 outside realistic range [0, 4] (1 instances)
- ca already satisfies > 0.444 (current: 1.0) (1 instances)
- ca already satisfies <= 0.328 (current: 0.0) (1 instances)

### chol
- chol already satisfies > 234.439 (current: 265.0) (1 instances)
- chol already satisfies > 239.782 (current: 256.0) (1 instances)
- chol already satisfies > 243.187 (current: 308.0) (1 instances)
- chol already satisfies > 247.976 (current: 258.0) (1 instances)

### cp
- cp already satisfies > 0.366 (current: 1.0) (1 instances)
- cp already satisfies <= 1.27 (current: 1.0) (1 instances)
- cp already satisfies > 1.173 (current: 2.0) (1 instances)
- cp target value -0.114 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 0.499 (current: 0.0) (1 instances)
- cp already satisfies > 0.007 (current: 2.0) (1 instances)
- cp already satisfies > 0.28 (current: 2.0) (1 instances)
- cp already satisfies <= 0.778 (current: 0.0) (1 instances)
- cp already satisfies <= 1.333 (current: 0.0) (1 instances)
- cp target value -0.567 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 0.833 (current: 0.0) (1 instances)
- cp already satisfies <= 0.301 (current: 0.0) (1 instances)
- cp target value -0.072 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 1.404 (current: 1.0) (1 instances)
- cp target value -0.314 outside realistic range [0, 3] (1 instances)

### exang
- exang target value -0.191 outside realistic range [0, 1] (1 instances)
- exang already satisfies > 0.932 (current: 1.0) (1 instances)
- exang already satisfies <= 0.144 (current: 0.0) (1 instances)
- exang already satisfies <= 0.096 (current: 0.0) (1 instances)
- exang already satisfies <= 0.167 (current: 0.0) (1 instances)
- exang already satisfies > 0.537 (current: 1.0) (1 instances)

### fbs
- fbs target value -0.263 outside realistic range [0, 1] (1 instances)
- fbs target value -0.209 outside realistic range [0, 1] (1 instances)
- fbs target value -0.103 outside realistic range [0, 1] (1 instances)
- fbs target value -0.418 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.056 (current: 0.0) (1 instances)

### oldpeak
- oldpeak target value -0.027 outside realistic range [0, 6.2] (1 instances)
- oldpeak target value -0.253 outside realistic range [0, 6.2] (1 instances)
- oldpeak target value -0.062 outside realistic range [0, 6.2] (1 instances)

### restecg
- restecg target value -0.38 outside realistic range [0, 2] (1 instances)
- restecg target value -0.416 outside realistic range [0, 2] (1 instances)
- restecg target value -0.186 outside realistic range [0, 2] (1 instances)
- restecg already satisfies > 0.211 (current: 1.0) (1 instances)
- restecg already satisfies <= 0.774 (current: 0.0) (1 instances)
- restecg target value -0.016 outside realistic range [0, 2] (1 instances)
- restecg already satisfies > 0.939 (current: 1.0) (1 instances)

### sex
- sex is immutable and cannot be changed (11 instances)

### slope
- slope already satisfies > 1.593 (current: 2.0) (1 instances)
- slope already satisfies > 0.86 (current: 1.0) (1 instances)
- slope already satisfies <= 1.431 (current: 1.0) (1 instances)
- slope already satisfies > 0.499 (current: 1.0) (1 instances)
- slope already satisfies > 0.64 (current: 1.0) (1 instances)
- slope already satisfies > 0.98 (current: 1.0) (1 instances)
- slope already satisfies > 0.538 (current: 1.0) (1 instances)
- slope already satisfies > 1.268 (current: 2.0) (1 instances)
- slope already satisfies <= 1.579 (current: 1.0) (1 instances)

### thal
- thal already satisfies <= 2.567 (current: 2.0) (1 instances)
- thal already satisfies > 2.732 (current: 3.0) (1 instances)
- thal already satisfies <= 2.435 (current: 2.0) (1 instances)
- thal already satisfies > 1.851 (current: 2.0) (1 instances)
- thal already satisfies > 2.448 (current: 3.0) (1 instances)
- thal already satisfies <= 2.918 (current: 2.0) (1 instances)
- thal target value 3.254 outside realistic range [0, 3] (1 instances)
- thal already satisfies > 1.861 (current: 3.0) (1 instances)
- thal already satisfies > 2.03 (current: 3.0) (1 instances)

### thalach
- thalach already satisfies > 132.881 (current: 172.0) (1 instances)
- thalach already satisfies > 140.166 (current: 173.0) (1 instances)
- thalach already satisfies > 142.477 (current: 166.0) (1 instances)
- thalach already satisfies > 156.998 (current: 162.0) (1 instances)
- thalach already satisfies <= 147.449 (current: 145.0) (1 instances)
- thalach already satisfies > 153.126 (current: 173.0) (1 instances)

### trestbps
- trestbps already satisfies > 112.22 (current: 120.0) (1 instances)
- trestbps already satisfies <= 152.639 (current: 122.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 195
- **Feasible conditions**: 91
- **Condition feasibility rate**: 46.7%

