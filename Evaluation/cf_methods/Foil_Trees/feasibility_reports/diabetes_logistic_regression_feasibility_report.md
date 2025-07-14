# Counterfactual Feasibility Analysis Report - DIABETES_LOGISTIC_REGRESSION

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 99
- **Feasible Counterfactuals**: 21
- **Feasibility Rate**: 21.2%

## Feature Constraint Analysis
- **Immutable features involved**: 8
- **Partially immutable features involved**: 21
- **Mutable features involved**: 149
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **Age**: 0/8 (0.0% feasible)
- **BMI**: 17/36 (47.2% feasible)
  - Average change magnitude: 9.411
  - Maximum change magnitude: 31.564
- **BloodPressure**: 3/13 (23.1% feasible)
  - Average change magnitude: 15.297
  - Maximum change magnitude: 23.985
- **DiabetesPedigreeFunction**: 4/12 (33.3% feasible)
  - Average change magnitude: 0.356
  - Maximum change magnitude: 0.964
- **Glucose**: 34/69 (49.3% feasible)
  - Average change magnitude: 24.333
  - Maximum change magnitude: 63.949
- **Insulin**: 4/9 (44.4% feasible)
  - Average change magnitude: 111.789
  - Maximum change magnitude: 149.995
- **Pregnancies**: 10/21 (47.6% feasible)
  - Average change magnitude: 3.709
  - Maximum change magnitude: 7.624
- **SkinThickness**: 3/10 (30.0% feasible)
  - Average change magnitude: 21.370
  - Maximum change magnitude: 25.682

## Common Feasibility Issues

### Age
- Age is immutable and cannot be changed (8 instances)

### BMI
- BMI already satisfies > 27.888 (current: 29.7) (1 instances)
- BMI already satisfies > 29.949 (current: 38.3) (1 instances)
- BMI already satisfies <= 40.099 (current: 31.1) (1 instances)
- BMI already satisfies <= 40.922 (current: 34.5) (1 instances)
- BMI already satisfies <= 26.234 (current: 22.2) (1 instances)
- BMI already satisfies <= 37.792 (current: 34.0) (1 instances)
- BMI already satisfies > 23.154 (current: 31.6) (1 instances)
- BMI already satisfies <= 39.67 (current: 30.1) (1 instances)
- BMI already satisfies > 33.139 (current: 35.8) (1 instances)
- BMI already satisfies <= 40.455 (current: 30.1) (1 instances)
- BMI already satisfies <= 42.434 (current: 25.4) (1 instances)
- BMI already satisfies > 26.429 (current: 29.7) (1 instances)
- BMI already satisfies <= 33.117 (current: 27.1) (1 instances)
- BMI already satisfies <= 36.654 (current: 28.8) (1 instances)
- BMI already satisfies <= 49.233 (current: 33.3) (1 instances)
- BMI already satisfies > 17.26 (current: 46.8) (1 instances)
- BMI already satisfies > 27.362 (current: 45.8) (1 instances)
- BMI already satisfies > 34.133 (current: 35.8) (1 instances)
- BMI already satisfies <= 44.174 (current: 30.9) (1 instances)

### BloodPressure
- BloodPressure already satisfies <= 84.532 (current: 80.0) (1 instances)
- BloodPressure already satisfies > 44.67 (current: 62.0) (1 instances)
- BloodPressure already satisfies > 57.096 (current: 72.0) (1 instances)
- BloodPressure already satisfies > 63.928 (current: 110.0) (1 instances)
- BloodPressure target value 38.426 outside realistic range [40, 200] (1 instances)
- BloodPressure target value 29.389 outside realistic range [40, 200] (1 instances)
- BloodPressure already satisfies > 46.319 (current: 86.0) (1 instances)
- BloodPressure already satisfies > 78.43 (current: 85.0) (1 instances)
- BloodPressure already satisfies > 59.779 (current: 104.0) (1 instances)
- BloodPressure already satisfies > 57.824 (current: 90.0) (1 instances)

### DiabetesPedigreeFunction
- DiabetesPedigreeFunction already satisfies <= 0.489 (current: 0.404) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.42 (current: 0.261) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.571 (current: 0.27) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.552 (current: 0.539) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.748 (current: 0.645) (1 instances)
- DiabetesPedigreeFunction target value -0.039 outside realistic range [0.08, 2.5] (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.667 (current: 0.294) (1 instances)
- DiabetesPedigreeFunction target value 0.026 outside realistic range [0.08, 2.5] (1 instances)

### Glucose
- Glucose already satisfies <= 133.439 (current: 74.0) (1 instances)
- Glucose already satisfies <= 146.338 (current: 62.0) (1 instances)
- Glucose already satisfies > 120.147 (current: 136.0) (1 instances)
- Glucose already satisfies <= 134.764 (current: 112.0) (1 instances)
- Glucose already satisfies <= 137.746 (current: 137.0) (1 instances)
- Glucose already satisfies <= 157.966 (current: 115.0) (1 instances)
- Glucose already satisfies <= 160.172 (current: 81.0) (1 instances)
- Glucose already satisfies <= 132.497 (current: 122.0) (1 instances)
- Glucose already satisfies <= 142.876 (current: 103.0) (1 instances)
- Glucose already satisfies <= 140.458 (current: 109.0) (1 instances)
- Glucose already satisfies <= 137.411 (current: 109.0) (1 instances)
- Glucose already satisfies > 128.362 (current: 154.0) (1 instances)
- Glucose already satisfies <= 151.362 (current: 115.0) (1 instances)
- Glucose already satisfies <= 150.329 (current: 113.0) (1 instances)
- Glucose already satisfies <= 141.475 (current: 111.0) (1 instances)
- Glucose already satisfies <= 161.186 (current: 120.0) (1 instances)
- Glucose already satisfies <= 175.874 (current: 158.0) (1 instances)
- Glucose already satisfies <= 143.871 (current: 106.0) (1 instances)
- Glucose already satisfies <= 155.263 (current: 74.0) (1 instances)
- Glucose already satisfies <= 140.013 (current: 137.0) (1 instances)
- Glucose already satisfies <= 153.024 (current: 84.0) (1 instances)
- Glucose already satisfies > 116.659 (current: 125.0) (1 instances)
- Glucose already satisfies <= 106.343 (current: 84.0) (1 instances)
- Glucose already satisfies <= 149.801 (current: 62.0) (1 instances)
- Glucose already satisfies > 113.22 (current: 140.0) (1 instances)
- Glucose already satisfies > 104.532 (current: 197.0) (1 instances)
- Glucose already satisfies <= 145.676 (current: 57.0) (1 instances)
- Glucose already satisfies <= 130.034 (current: 130.0) (1 instances)
- Glucose already satisfies <= 151.763 (current: 114.0) (1 instances)
- Glucose already satisfies <= 145.429 (current: 108.0) (1 instances)
- Glucose already satisfies <= 134.629 (current: 112.0) (1 instances)
- Glucose already satisfies > 126.679 (current: 158.0) (1 instances)
- Glucose already satisfies <= 134.554 (current: 124.0) (1 instances)
- Glucose already satisfies <= 143.56 (current: 102.0) (1 instances)
- Glucose already satisfies <= 131.173 (current: 120.0) (1 instances)

### Insulin
- Insulin target value -82.701 outside realistic range [14, 900] (1 instances)
- Insulin target value -70.168 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 85.882 (current: 51.0) (1 instances)
- Insulin target value 7.414 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 83.747 (current: 0.0) (1 instances)

### Pregnancies
- Pregnancies already satisfies > 1.943 (current: 2.0) (1 instances)
- Pregnancies already satisfies > 5.521 (current: 6.0) (1 instances)
- Pregnancies already satisfies <= 9.893 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 7.557 (current: 7.0) (1 instances)
- Pregnancies target value -1.268 outside realistic range [0, 20] (1 instances)
- Pregnancies already satisfies <= 2.611 (current: 2.0) (1 instances)
- Pregnancies already satisfies > 1.957 (current: 4.0) (1 instances)
- Pregnancies already satisfies <= 5.077 (current: 0.0) (1 instances)
- Pregnancies already satisfies <= 5.102 (current: 2.0) (1 instances)
- Pregnancies already satisfies <= 6.664 (current: 3.0) (1 instances)
- Pregnancies already satisfies <= 6.143 (current: 3.0) (1 instances)

### SkinThickness
- SkinThickness target value 6.103 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 20.032 (current: 0.0) (1 instances)
- SkinThickness target value 0.317 outside realistic range [7, 100] (1 instances)
- SkinThickness target value 4.176 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 14.293 (current: 0.0) (1 instances)
- SkinThickness already satisfies > 22.508 (current: 37.0) (1 instances)
- SkinThickness target value -10.584 outside realistic range [7, 100] (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 178
- **Feasible conditions**: 75
- **Condition feasibility rate**: 42.1%

