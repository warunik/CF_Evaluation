# Counterfactual Feasibility Analysis Report - HEART_MLP

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 119
- **Feasible Counterfactuals**: 26
- **Feasibility Rate**: 21.8%

## Feature Constraint Analysis
- **Immutable features involved**: 25
- **Partially immutable features involved**: 115
- **Mutable features involved**: 67
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **age**: 0/9 (0.0% feasible)
- **ca**: 16/38 (42.1% feasible)
  - Average change magnitude: 1.214
  - Maximum change magnitude: 2.925
- **chol**: 7/11 (63.6% feasible)
  - Average change magnitude: 34.048
  - Maximum change magnitude: 81.806
- **cp**: 21/40 (52.5% feasible)
  - Average change magnitude: 0.750
  - Maximum change magnitude: 1.547
- **exang**: 5/11 (45.5% feasible)
  - Average change magnitude: 0.747
  - Maximum change magnitude: 0.949
- **fbs**: 4/10 (40.0% feasible)
  - Average change magnitude: 0.607
  - Maximum change magnitude: 0.891
- **oldpeak**: 6/13 (46.2% feasible)
  - Average change magnitude: 1.491
  - Maximum change magnitude: 2.880
- **restecg**: 3/7 (42.9% feasible)
  - Average change magnitude: 0.775
  - Maximum change magnitude: 0.929
- **sex**: 0/16 (0.0% feasible)
- **slope**: 12/18 (66.7% feasible)
  - Average change magnitude: 0.506
  - Maximum change magnitude: 1.097
- **thal**: 5/12 (41.7% feasible)
  - Average change magnitude: 0.566
  - Maximum change magnitude: 0.825
- **thalach**: 4/11 (36.4% feasible)
  - Average change magnitude: 29.131
  - Maximum change magnitude: 67.808
- **trestbps**: 3/11 (27.3% feasible)
  - Average change magnitude: 17.263
  - Maximum change magnitude: 23.381

## Common Feasibility Issues

### age
- age is immutable and cannot be changed (9 instances)

### ca
- ca already satisfies <= 0.566 (current: 0.0) (1 instances)
- ca already satisfies <= 0.779 (current: 0.0) (1 instances)
- ca already satisfies <= 1.209 (current: 1.0) (1 instances)
- ca target value -0.335 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 0.713 (current: 0.0) (1 instances)
- ca target value -2.285 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 0.125 (current: 0.0) (1 instances)
- ca already satisfies <= 0.426 (current: 0.0) (1 instances)
- ca already satisfies > 1.174 (current: 2.0) (1 instances)
- ca already satisfies > 0.703 (current: 1.0) (1 instances)
- ca target value -0.21 outside realistic range [0, 4] (1 instances)
- ca target value -0.925 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 1.149 (current: 0.0) (1 instances)
- ca already satisfies <= 1.038 (current: 1.0) (1 instances)
- ca already satisfies <= 1.228 (current: 0.0) (1 instances)
- ca target value -0.999 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 1.258 (current: 0.0) (1 instances)
- ca already satisfies <= 1.099 (current: 1.0) (1 instances)
- ca target value -0.107 outside realistic range [0, 4] (1 instances)
- ca already satisfies <= 1.66 (current: 0.0) (1 instances)
- ca already satisfies <= 1.126 (current: 0.0) (1 instances)
- ca already satisfies <= 0.117 (current: 0.0) (1 instances)

### chol
- chol already satisfies > 210.044 (current: 212.0) (1 instances)
- chol already satisfies > 196.105 (current: 250.0) (1 instances)
- chol already satisfies > 183.041 (current: 342.0) (1 instances)
- chol already satisfies > 276.557 (current: 288.0) (1 instances)

### cp
- cp already satisfies > 0.029 (current: 2.0) (1 instances)
- cp already satisfies <= 0.954 (current: 0.0) (1 instances)
- cp already satisfies <= 1.237 (current: 0.0) (1 instances)
- cp target value -0.125 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 2.069 (current: 2.0) (1 instances)
- cp already satisfies > 0.271 (current: 2.0) (1 instances)
- cp already satisfies <= 1.214 (current: 1.0) (1 instances)
- cp already satisfies <= 1.043 (current: 1.0) (1 instances)
- cp already satisfies <= 0.949 (current: 0.0) (1 instances)
- cp already satisfies <= 1.745 (current: 1.0) (1 instances)
- cp target value -0.591 outside realistic range [0, 3] (1 instances)
- cp already satisfies <= 0.02 (current: 0.0) (1 instances)
- cp already satisfies > 0.784 (current: 2.0) (1 instances)
- cp already satisfies <= 0.819 (current: 0.0) (1 instances)
- cp already satisfies <= 1.958 (current: 0.0) (1 instances)
- cp already satisfies > 0.685 (current: 3.0) (1 instances)
- cp target value -0.663 outside realistic range [0, 3] (1 instances)
- cp already satisfies > 0.389 (current: 2.0) (1 instances)
- cp already satisfies > 1.654 (current: 2.0) (1 instances)

### exang
- exang already satisfies > 0.278 (current: 1.0) (1 instances)
- exang target value -0.426 outside realistic range [0, 1] (1 instances)
- exang already satisfies <= 0.288 (current: 0.0) (1 instances)
- exang already satisfies > 0.308 (current: 1.0) (1 instances)
- exang target value -0.278 outside realistic range [0, 1] (1 instances)
- exang target value -0.225 outside realistic range [0, 1] (1 instances)

### fbs
- fbs target value -0.529 outside realistic range [0, 1] (1 instances)
- fbs target value -0.324 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.157 (current: 0.0) (1 instances)
- fbs target value -0.255 outside realistic range [0, 1] (1 instances)
- fbs already satisfies <= 0.05 (current: 0.0) (1 instances)
- fbs target value -0.583 outside realistic range [0, 1] (1 instances)

### oldpeak
- oldpeak target value -1.316 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies <= 1.329 (current: 0.9) (1 instances)
- oldpeak target value -0.116 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies <= 1.924 (current: 0.8) (1 instances)
- oldpeak target value -0.274 outside realistic range [0, 6.2] (1 instances)
- oldpeak already satisfies <= 1.717 (current: 0.0) (1 instances)
- oldpeak already satisfies > 0.432 (current: 0.8) (1 instances)

### restecg
- restecg already satisfies > 0.272 (current: 1.0) (1 instances)
- restecg target value -0.221 outside realistic range [0, 2] (1 instances)
- restecg target value -0.036 outside realistic range [0, 2] (1 instances)
- restecg already satisfies <= 0.073 (current: 0.0) (1 instances)

### sex
- sex is immutable and cannot be changed (16 instances)

### slope
- slope already satisfies <= 1.546 (current: 1.0) (1 instances)
- slope already satisfies > 0.77 (current: 1.0) (1 instances)
- slope already satisfies > 0.558 (current: 1.0) (1 instances)
- slope already satisfies > 1.302 (current: 2.0) (1 instances)
- slope already satisfies <= 1.825 (current: 1.0) (1 instances)
- slope already satisfies > 1.742 (current: 2.0) (1 instances)

### thal
- thal already satisfies > 1.372 (current: 3.0) (1 instances)
- thal already satisfies > 2.176 (current: 3.0) (1 instances)
- thal already satisfies <= 2.034 (current: 2.0) (1 instances)
- thal target value 3.01 outside realistic range [0, 3] (1 instances)
- thal already satisfies > 1.904 (current: 2.0) (1 instances)
- thal already satisfies <= 2.554 (current: 2.0) (1 instances)
- thal already satisfies > 2.441 (current: 3.0) (1 instances)

### thalach
- thalach already satisfies <= 164.068 (current: 118.0) (1 instances)
- thalach already satisfies <= 155.819 (current: 136.0) (1 instances)
- thalach already satisfies <= 178.792 (current: 132.0) (1 instances)
- thalach already satisfies > 126.575 (current: 160.0) (1 instances)
- thalach already satisfies <= 162.149 (current: 156.0) (1 instances)
- thalach already satisfies <= 135.778 (current: 122.0) (1 instances)
- thalach already satisfies > 140.802 (current: 173.0) (1 instances)

### trestbps
- trestbps already satisfies > 117.119 (current: 170.0) (1 instances)
- trestbps already satisfies <= 136.632 (current: 102.0) (1 instances)
- trestbps already satisfies > 108.488 (current: 170.0) (1 instances)
- trestbps already satisfies > 132.519 (current: 170.0) (1 instances)
- trestbps already satisfies > 137.711 (current: 140.0) (1 instances)
- trestbps already satisfies <= 143.883 (current: 108.0) (1 instances)
- trestbps already satisfies > 115.716 (current: 130.0) (1 instances)
- trestbps already satisfies <= 141.644 (current: 138.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 207
- **Feasible conditions**: 86
- **Condition feasibility rate**: 41.5%

