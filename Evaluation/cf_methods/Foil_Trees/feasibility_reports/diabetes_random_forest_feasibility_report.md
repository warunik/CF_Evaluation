# Counterfactual Feasibility Analysis Report - DIABETES_RANDOM_FOREST

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 83
- **Feasible Counterfactuals**: 14
- **Feasibility Rate**: 16.9%

## Feature Constraint Analysis
- **Immutable features involved**: 16
- **Partially immutable features involved**: 22
- **Mutable features involved**: 127
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **Age**: 0/16 (0.0% feasible)
- **BMI**: 12/22 (54.5% feasible)
  - Average change magnitude: 8.056
  - Maximum change magnitude: 28.090
- **BloodPressure**: 2/9 (22.2% feasible)
  - Average change magnitude: 7.333
  - Maximum change magnitude: 10.250
- **DiabetesPedigreeFunction**: 6/23 (26.1% feasible)
  - Average change magnitude: 0.214
  - Maximum change magnitude: 0.575
- **Glucose**: 28/50 (56.0% feasible)
  - Average change magnitude: 31.228
  - Maximum change magnitude: 68.585
- **Insulin**: 3/17 (17.6% feasible)
  - Average change magnitude: 137.412
  - Maximum change magnitude: 170.771
- **Pregnancies**: 9/22 (40.9% feasible)
  - Average change magnitude: 2.581
  - Maximum change magnitude: 5.441
- **SkinThickness**: 1/6 (16.7% feasible)
  - Average change magnitude: 0.281
  - Maximum change magnitude: 0.281

## Common Feasibility Issues

### Age
- Age is immutable and cannot be changed (16 instances)

### BMI
- BMI already satisfies <= 36.821 (current: 31.6) (1 instances)
- BMI already satisfies <= 30.506 (current: 27.4) (1 instances)
- BMI already satisfies <= 29.919 (current: 27.6) (1 instances)
- BMI already satisfies > 30.106 (current: 50.0) (1 instances)
- BMI already satisfies > 27.271 (current: 45.2) (1 instances)
- BMI already satisfies <= 31.883 (current: 28.8) (1 instances)
- BMI already satisfies <= 31.716 (current: 30.4) (1 instances)
- BMI already satisfies > 27.161 (current: 29.6) (1 instances)
- BMI already satisfies > 30.068 (current: 40.6) (1 instances)
- BMI already satisfies <= 36.702 (current: 29.5) (1 instances)

### BloodPressure
- BloodPressure already satisfies <= 68.778 (current: 58.0) (1 instances)
- BloodPressure already satisfies <= 82.753 (current: 60.0) (1 instances)
- BloodPressure already satisfies > 59.404 (current: 78.0) (1 instances)
- BloodPressure already satisfies > 53.058 (current: 64.0) (1 instances)
- BloodPressure already satisfies <= 88.861 (current: 68.0) (1 instances)
- BloodPressure already satisfies > 61.042 (current: 84.0) (1 instances)
- BloodPressure already satisfies > 59.314 (current: 82.0) (1 instances)

### DiabetesPedigreeFunction
- DiabetesPedigreeFunction already satisfies <= 0.699 (current: 0.205) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.446 (current: 0.325) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.189 (current: 0.457) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.728 (current: 0.627) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.156 (current: 0.342) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 1.073 (current: 0.956) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.344 (current: 0.254) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.243 (current: 0.452) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.525 (current: 0.26) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.779 (current: 0.227) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.583 (current: 1.781) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.333 (current: 0.514) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.154 (current: 0.451) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.814 (current: 0.167) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 1.024 (current: 0.15) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.616 (current: 0.19) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.647 (current: 0.27) (1 instances)

### Glucose
- Glucose already satisfies <= 132.272 (current: 103.0) (1 instances)
- Glucose already satisfies <= 142.876 (current: 103.0) (1 instances)
- Glucose already satisfies <= 141.452 (current: 111.0) (1 instances)
- Glucose already satisfies > 139.742 (current: 142.0) (1 instances)
- Glucose already satisfies > 109.11 (current: 141.0) (1 instances)
- Glucose already satisfies <= 147.875 (current: 141.0) (1 instances)
- Glucose already satisfies > 145.299 (current: 166.0) (1 instances)
- Glucose already satisfies <= 129.055 (current: 112.0) (1 instances)
- Glucose already satisfies <= 154.866 (current: 57.0) (1 instances)
- Glucose already satisfies <= 159.379 (current: 119.0) (1 instances)
- Glucose already satisfies <= 108.0 (current: 85.0) (1 instances)
- Glucose already satisfies > 123.298 (current: 136.0) (1 instances)
- Glucose already satisfies > 132.023 (current: 144.0) (1 instances)
- Glucose already satisfies <= 131.625 (current: 117.0) (1 instances)
- Glucose already satisfies <= 140.177 (current: 102.0) (1 instances)
- Glucose already satisfies > 136.235 (current: 177.0) (1 instances)
- Glucose already satisfies > 122.702 (current: 153.0) (1 instances)
- Glucose already satisfies <= 141.809 (current: 119.0) (1 instances)
- Glucose already satisfies <= 139.97 (current: 112.0) (1 instances)
- Glucose already satisfies <= 150.785 (current: 99.0) (1 instances)
- Glucose already satisfies > 111.95 (current: 189.0) (1 instances)
- Glucose already satisfies > 108.495 (current: 134.0) (1 instances)

### Insulin
- Insulin target value -77.571 outside realistic range [14, 900] (1 instances)
- Insulin target value 5.383 outside realistic range [14, 900] (1 instances)
- Insulin target value -61.885 outside realistic range [14, 900] (1 instances)
- Insulin target value -9.306 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies > 53.204 (current: 128.0) (1 instances)
- Insulin target value -9.956 outside realistic range [14, 900] (1 instances)
- Insulin target value 7.414 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 110.456 (current: 0.0) (1 instances)
- Insulin already satisfies <= 169.645 (current: 130.0) (1 instances)
- Insulin already satisfies > 150.568 (current: 478.0) (1 instances)
- Insulin target value -0.302 outside realistic range [14, 900] (1 instances)
- Insulin target value -27.73 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 200.193 (current: 0.0) (1 instances)
- Insulin already satisfies <= 136.649 (current: 0.0) (1 instances)

### Pregnancies
- Pregnancies already satisfies <= 10.204 (current: 1.0) (1 instances)
- Pregnancies already satisfies <= 9.531 (current: 4.0) (1 instances)
- Pregnancies already satisfies <= 6.407 (current: 4.0) (1 instances)
- Pregnancies already satisfies <= 5.781 (current: 0.0) (1 instances)
- Pregnancies already satisfies <= 6.171 (current: 3.0) (1 instances)
- Pregnancies already satisfies <= 2.602 (current: 2.0) (1 instances)
- Pregnancies already satisfies > 0.397 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 7.704 (current: 0.0) (1 instances)
- Pregnancies already satisfies <= 6.063 (current: 6.0) (1 instances)
- Pregnancies already satisfies <= 7.119 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 6.995 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 6.227 (current: 4.0) (1 instances)
- Pregnancies already satisfies <= 7.159 (current: 5.0) (1 instances)

### SkinThickness
- SkinThickness target value -1.787 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 13.363 (current: 0.0) (1 instances)
- SkinThickness already satisfies <= 44.258 (current: 30.0) (1 instances)
- SkinThickness already satisfies <= 39.897 (current: 33.0) (1 instances)
- SkinThickness already satisfies <= 19.248 (current: 0.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 165
- **Feasible conditions**: 61
- **Condition feasibility rate**: 37.0%

