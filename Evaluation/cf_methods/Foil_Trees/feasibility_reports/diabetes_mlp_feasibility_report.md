# Counterfactual Feasibility Analysis Report - DIABETES_MLP

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 93
- **Feasible Counterfactuals**: 21
- **Feasibility Rate**: 22.6%

## Feature Constraint Analysis
- **Immutable features involved**: 8
- **Partially immutable features involved**: 21
- **Mutable features involved**: 140
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **Age**: 0/8 (0.0% feasible)
- **BMI**: 13/35 (37.1% feasible)
  - Average change magnitude: 5.775
  - Maximum change magnitude: 12.453
- **BloodPressure**: 3/9 (33.3% feasible)
  - Average change magnitude: 41.996
  - Maximum change magnitude: 62.663
- **DiabetesPedigreeFunction**: 5/18 (27.8% feasible)
  - Average change magnitude: 0.215
  - Maximum change magnitude: 0.444
- **Glucose**: 25/47 (53.2% feasible)
  - Average change magnitude: 25.521
  - Maximum change magnitude: 61.631
- **Insulin**: 4/14 (28.6% feasible)
  - Average change magnitude: 49.045
  - Maximum change magnitude: 133.513
- **Pregnancies**: 10/21 (47.6% feasible)
  - Average change magnitude: 3.602
  - Maximum change magnitude: 10.624
- **SkinThickness**: 3/17 (17.6% feasible)
  - Average change magnitude: 2.931
  - Maximum change magnitude: 4.198

## Common Feasibility Issues

### Age
- Age is immutable and cannot be changed (8 instances)

### BMI
- BMI already satisfies <= 36.589 (current: 28.8) (1 instances)
- BMI already satisfies > 25.553 (current: 40.2) (1 instances)
- BMI already satisfies > 39.969 (current: 43.3) (1 instances)
- BMI already satisfies <= 32.771 (current: 25.2) (1 instances)
- BMI already satisfies <= 50.27 (current: 33.2) (1 instances)
- BMI already satisfies > 23.154 (current: 31.6) (1 instances)
- BMI already satisfies <= 39.67 (current: 30.1) (1 instances)
- BMI already satisfies > 25.478 (current: 42.0) (1 instances)
- BMI target value 13.521 outside realistic range [15, 60] (1 instances)
- BMI already satisfies <= 33.014 (current: 27.9) (1 instances)
- BMI already satisfies <= 41.134 (current: 36.7) (1 instances)
- BMI already satisfies <= 31.653 (current: 31.2) (1 instances)
- BMI already satisfies <= 29.096 (current: 24.3) (1 instances)
- BMI already satisfies <= 35.924 (current: 30.0) (1 instances)
- BMI already satisfies <= 42.576 (current: 29.6) (1 instances)
- BMI already satisfies <= 35.107 (current: 33.6) (1 instances)
- BMI already satisfies <= 38.711 (current: 28.9) (1 instances)
- BMI already satisfies <= 35.519 (current: 27.1) (1 instances)
- BMI already satisfies <= 43.299 (current: 32.9) (1 instances)
- BMI already satisfies <= 29.268 (current: 27.6) (1 instances)
- BMI already satisfies > 27.496 (current: 34.9) (1 instances)
- BMI already satisfies > 17.284 (current: 36.9) (1 instances)

### BloodPressure
- BloodPressure already satisfies > 50.236 (current: 52.0) (1 instances)
- BloodPressure target value 37.376 outside realistic range [40, 200] (1 instances)
- BloodPressure target value 38.977 outside realistic range [40, 200] (1 instances)
- BloodPressure already satisfies > 73.223 (current: 108.0) (1 instances)
- BloodPressure already satisfies <= 85.694 (current: 58.0) (1 instances)
- BloodPressure already satisfies > 51.233 (current: 72.0) (1 instances)

### DiabetesPedigreeFunction
- DiabetesPedigreeFunction already satisfies > 0.338 (current: 0.578) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.298 (current: 0.218) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.53 (current: 0.209) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.74 (current: 0.652) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.691 (current: 0.167) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.7 (current: 0.204) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.151 (current: 0.721) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.495 (current: 0.294) (1 instances)
- DiabetesPedigreeFunction target value -0.377 outside realistic range [0.08, 2.5] (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.908 (current: 0.22) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.501 (current: 0.411) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.404 (current: 1.781) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.38 (current: 0.452) (1 instances)

### Glucose
- Glucose already satisfies <= 134.764 (current: 112.0) (1 instances)
- Glucose already satisfies <= 144.11 (current: 85.0) (1 instances)
- Glucose already satisfies <= 121.833 (current: 109.0) (1 instances)
- Glucose already satisfies <= 142.876 (current: 103.0) (1 instances)
- Glucose already satisfies <= 122.922 (current: 109.0) (1 instances)
- Glucose already satisfies <= 150.695 (current: 117.0) (1 instances)
- Glucose already satisfies <= 138.666 (current: 137.0) (1 instances)
- Glucose already satisfies <= 134.023 (current: 125.0) (1 instances)
- Glucose already satisfies <= 139.46 (current: 122.0) (1 instances)
- Glucose already satisfies <= 164.688 (current: 119.0) (1 instances)
- Glucose already satisfies > 120.358 (current: 154.0) (1 instances)
- Glucose already satisfies > 112.354 (current: 134.0) (1 instances)
- Glucose already satisfies <= 142.892 (current: 106.0) (1 instances)
- Glucose already satisfies <= 115.86 (current: 102.0) (1 instances)
- Glucose already satisfies <= 128.045 (current: 97.0) (1 instances)
- Glucose already satisfies <= 152.039 (current: 112.0) (1 instances)
- Glucose already satisfies <= 131.794 (current: 118.0) (1 instances)
- Glucose already satisfies > 126.133 (current: 183.0) (1 instances)
- Glucose already satisfies <= 158.96 (current: 97.0) (1 instances)
- Glucose already satisfies > 119.17 (current: 179.0) (1 instances)
- Glucose already satisfies <= 163.598 (current: 88.0) (1 instances)
- Glucose already satisfies <= 125.61 (current: 122.0) (1 instances)

### Insulin
- Insulin already satisfies <= 122.817 (current: 105.0) (1 instances)
- Insulin already satisfies <= 158.762 (current: 0.0) (1 instances)
- Insulin already satisfies > 138.899 (current: 495.0) (1 instances)
- Insulin already satisfies > 21.13 (current: 228.0) (1 instances)
- Insulin already satisfies <= 269.472 (current: 168.0) (1 instances)
- Insulin target value 12.184 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 250.51 (current: 0.0) (1 instances)
- Insulin already satisfies > 78.584 (current: 99.0) (1 instances)
- Insulin target value -91.535 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies <= 85.383 (current: 0.0) (1 instances)

### Pregnancies
- Pregnancies already satisfies > 1.193 (current: 5.0) (1 instances)
- Pregnancies already satisfies > 3.565 (current: 5.0) (1 instances)
- Pregnancies target value -1.661 outside realistic range [0, 20] (1 instances)
- Pregnancies already satisfies > 1.711 (current: 4.0) (1 instances)
- Pregnancies already satisfies <= 9.893 (current: 5.0) (1 instances)
- Pregnancies target value -0.412 outside realistic range [0, 20] (1 instances)
- Pregnancies target value -3.113 outside realistic range [0, 20] (1 instances)
- Pregnancies already satisfies <= 5.183 (current: 4.0) (1 instances)
- Pregnancies already satisfies > 3.932 (current: 9.0) (1 instances)
- Pregnancies already satisfies > 2.878 (current: 5.0) (1 instances)
- Pregnancies already satisfies <= 4.826 (current: 0.0) (1 instances)

### SkinThickness
- SkinThickness target value -0.841 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies > 27.161 (current: 36.0) (1 instances)
- SkinThickness target value -6.741 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 20.482 (current: 0.0) (1 instances)
- SkinThickness already satisfies > 19.783 (current: 25.0) (1 instances)
- SkinThickness already satisfies > 17.461 (current: 35.0) (1 instances)
- SkinThickness already satisfies > 21.69 (current: 27.0) (1 instances)
- SkinThickness already satisfies > 28.815 (current: 29.0) (1 instances)
- SkinThickness already satisfies <= 14.293 (current: 0.0) (1 instances)
- SkinThickness already satisfies > 25.066 (current: 41.0) (1 instances)
- SkinThickness already satisfies > 7.44 (current: 27.0) (1 instances)
- SkinThickness target value -7.816 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 23.771 (current: 0.0) (1 instances)
- SkinThickness already satisfies > 26.224 (current: 39.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 169
- **Feasible conditions**: 63
- **Condition feasibility rate**: 37.3%

