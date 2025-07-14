# Counterfactual Feasibility Analysis Report - DIABETES_DECISION_TREE

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 73
- **Feasible Counterfactuals**: 17
- **Feasibility Rate**: 23.3%

## Feature Constraint Analysis
- **Immutable features involved**: 11
- **Partially immutable features involved**: 12
- **Mutable features involved**: 104
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **Age**: 0/11 (0.0% feasible)
- **BMI**: 9/20 (45.0% feasible)
  - Average change magnitude: 5.034
  - Maximum change magnitude: 8.683
- **BloodPressure**: 4/14 (28.6% feasible)
  - Average change magnitude: 9.783
  - Maximum change magnitude: 15.543
- **DiabetesPedigreeFunction**: 6/14 (42.9% feasible)
  - Average change magnitude: 0.424
  - Maximum change magnitude: 1.846
- **Glucose**: 13/34 (38.2% feasible)
  - Average change magnitude: 25.076
  - Maximum change magnitude: 66.717
- **Insulin**: 3/11 (27.3% feasible)
  - Average change magnitude: 358.758
  - Maximum change magnitude: 729.386
- **Pregnancies**: 5/12 (41.7% feasible)
  - Average change magnitude: 1.547
  - Maximum change magnitude: 3.048
- **SkinThickness**: 3/11 (27.3% feasible)
  - Average change magnitude: 19.620
  - Maximum change magnitude: 36.999

## Common Feasibility Issues

### Age
- Age is immutable and cannot be changed (11 instances)

### BMI
- BMI already satisfies > 29.838 (current: 43.2) (1 instances)
- BMI already satisfies <= 29.761 (current: 24.7) (1 instances)
- BMI already satisfies > 29.787 (current: 30.1) (1 instances)
- BMI already satisfies > 31.902 (current: 32.9) (1 instances)
- BMI already satisfies <= 37.83 (current: 36.7) (1 instances)
- BMI already satisfies > 24.273 (current: 34.5) (1 instances)
- BMI already satisfies > 26.675 (current: 39.6) (1 instances)
- BMI already satisfies <= 34.735 (current: 27.1) (1 instances)
- BMI already satisfies <= 40.925 (current: 27.6) (1 instances)
- BMI already satisfies > 28.252 (current: 31.1) (1 instances)
- BMI already satisfies > 27.019 (current: 31.0) (1 instances)

### BloodPressure
- BloodPressure target value 17.842 outside realistic range [40, 200] (1 instances)
- BloodPressure already satisfies <= 71.297 (current: 68.0) (1 instances)
- BloodPressure already satisfies > 59.678 (current: 76.0) (1 instances)
- BloodPressure already satisfies <= 92.38 (current: 60.0) (1 instances)
- BloodPressure already satisfies <= 82.417 (current: 74.0) (1 instances)
- BloodPressure target value 35.196 outside realistic range [40, 200] (1 instances)
- BloodPressure already satisfies > 58.853 (current: 70.0) (1 instances)
- BloodPressure already satisfies <= 84.583 (current: 0.0) (1 instances)
- BloodPressure already satisfies > 42.485 (current: 76.0) (1 instances)
- BloodPressure already satisfies <= 89.396 (current: 76.0) (1 instances)

### DiabetesPedigreeFunction
- DiabetesPedigreeFunction target value -0.148 outside realistic range [0.08, 2.5] (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.123 (current: 0.375) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.199 (current: 0.388) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.641 (current: 0.294) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.127 (current: 0.355) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.145 (current: 0.262) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.441 (current: 0.962) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.362 (current: 0.294) (1 instances)

### Glucose
- Glucose already satisfies > 131.376 (current: 142.0) (1 instances)
- Glucose already satisfies <= 130.894 (current: 125.0) (1 instances)
- Glucose already satisfies <= 158.779 (current: 125.0) (1 instances)
- Glucose already satisfies <= 148.921 (current: 106.0) (1 instances)
- Glucose already satisfies <= 149.953 (current: 62.0) (1 instances)
- Glucose already satisfies > 154.953 (current: 158.0) (1 instances)
- Glucose already satisfies > 143.63 (current: 144.0) (1 instances)
- Glucose already satisfies <= 134.688 (current: 118.0) (1 instances)
- Glucose already satisfies <= 155.713 (current: 99.0) (1 instances)
- Glucose already satisfies <= 151.122 (current: 129.0) (1 instances)
- Glucose already satisfies > 128.256 (current: 154.0) (1 instances)
- Glucose already satisfies <= 138.014 (current: 134.0) (1 instances)
- Glucose already satisfies <= 135.626 (current: 84.0) (1 instances)
- Glucose already satisfies <= 132.168 (current: 105.0) (1 instances)
- Glucose already satisfies > 122.838 (current: 179.0) (1 instances)
- Glucose already satisfies <= 152.549 (current: 131.0) (1 instances)
- Glucose already satisfies > 140.561 (current: 146.0) (1 instances)
- Glucose already satisfies <= 140.649 (current: 112.0) (1 instances)
- Glucose already satisfies > 103.657 (current: 177.0) (1 instances)
- Glucose already satisfies <= 141.169 (current: 115.0) (1 instances)
- Glucose already satisfies > 126.679 (current: 158.0) (1 instances)

### Insulin
- Insulin target value -18.466 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 211.798 (current: 0.0) (1 instances)
- Insulin already satisfies > 25.732 (current: 32.0) (1 instances)
- Insulin target value -131.539 outside realistic range [14, 900] (1 instances)
- Insulin target value -38.425 outside realistic range [14, 900] (1 instances)
- Insulin target value 1.327 outside realistic range [14, 900] (1 instances)
- Insulin target value -31.638 outside realistic range [14, 900] (1 instances)
- Insulin target value -89.266 outside realistic range [14, 900] (1 instances)

### Pregnancies
- Pregnancies already satisfies <= 8.629 (current: 0.0) (1 instances)
- Pregnancies already satisfies <= 6.804 (current: 6.0) (1 instances)
- Pregnancies already satisfies > 4.46 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 6.749 (current: 2.0) (1 instances)
- Pregnancies target value -1.278 outside realistic range [0, 20] (1 instances)
- Pregnancies already satisfies <= 4.004 (current: 2.0) (1 instances)
- Pregnancies already satisfies <= 8.472 (current: 0.0) (1 instances)

### SkinThickness
- SkinThickness already satisfies > 26.755 (current: 35.0) (1 instances)
- SkinThickness target value 4.165 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 37.122 (current: 0.0) (1 instances)
- SkinThickness already satisfies > 27.841 (current: 35.0) (1 instances)
- SkinThickness already satisfies <= 40.948 (current: 30.0) (1 instances)
- SkinThickness already satisfies <= 14.293 (current: 0.0) (1 instances)
- SkinThickness target value 2.698 outside realistic range [7, 100] (1 instances)
- SkinThickness target value 5.554 outside realistic range [7, 100] (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 127
- **Feasible conditions**: 43
- **Condition feasibility rate**: 33.9%

