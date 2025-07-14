# Counterfactual Feasibility Analysis Report - DIABETES_XGBOOST

## Executive Summary
- **Total Successful Counterfactuals Analyzed**: 90
- **Feasible Counterfactuals**: 14
- **Feasibility Rate**: 15.6%

## Feature Constraint Analysis
- **Immutable features involved**: 19
- **Partially immutable features involved**: 15
- **Mutable features involved**: 128
- **Unknown features involved**: 0

## Feature-wise Feasibility
- **Age**: 0/19 (0.0% feasible)
- **BMI**: 10/22 (45.5% feasible)
  - Average change magnitude: 6.147
  - Maximum change magnitude: 10.047
- **BloodPressure**: 3/11 (27.3% feasible)
  - Average change magnitude: 23.569
  - Maximum change magnitude: 41.034
- **DiabetesPedigreeFunction**: 4/16 (25.0% feasible)
  - Average change magnitude: 0.225
  - Maximum change magnitude: 0.585
- **Glucose**: 21/51 (41.2% feasible)
  - Average change magnitude: 35.998
  - Maximum change magnitude: 76.649
- **Insulin**: 5/10 (50.0% feasible)
  - Average change magnitude: 105.337
  - Maximum change magnitude: 148.737
- **Pregnancies**: 7/15 (46.7% feasible)
  - Average change magnitude: 2.231
  - Maximum change magnitude: 7.530
- **SkinThickness**: 9/18 (50.0% feasible)
  - Average change magnitude: 10.590
  - Maximum change magnitude: 29.684

## Common Feasibility Issues

### Age
- Age is immutable and cannot be changed (19 instances)

### BMI
- BMI already satisfies <= 36.821 (current: 31.6) (1 instances)
- BMI already satisfies <= 45.362 (current: 33.3) (1 instances)
- BMI already satisfies > 30.719 (current: 45.4) (1 instances)
- BMI already satisfies <= 30.16 (current: 27.4) (1 instances)
- BMI already satisfies > 23.154 (current: 31.6) (1 instances)
- BMI already satisfies > 25.478 (current: 42.0) (1 instances)
- BMI already satisfies <= 33.987 (current: 31.0) (1 instances)
- BMI already satisfies <= 39.91 (current: 29.7) (1 instances)
- BMI already satisfies > 31.799 (current: 42.7) (1 instances)
- BMI already satisfies > 27.262 (current: 36.9) (1 instances)
- BMI already satisfies > 20.013 (current: 29.7) (1 instances)
- BMI already satisfies <= 28.829 (current: 22.7) (1 instances)

### BloodPressure
- BloodPressure already satisfies <= 84.725 (current: 60.0) (1 instances)
- BloodPressure already satisfies > 78.89 (current: 95.0) (1 instances)
- BloodPressure already satisfies > 72.221 (current: 104.0) (1 instances)
- BloodPressure already satisfies > 52.85 (current: 78.0) (1 instances)
- BloodPressure already satisfies > 57.11 (current: 76.0) (1 instances)
- BloodPressure already satisfies > 62.984 (current: 66.0) (1 instances)
- BloodPressure target value 18.183 outside realistic range [40, 200] (1 instances)
- BloodPressure already satisfies > 71.265 (current: 72.0) (1 instances)

### DiabetesPedigreeFunction
- DiabetesPedigreeFunction already satisfies > 0.083 (current: 0.572) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.75 (current: 0.375) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.243 (current: 0.452) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.676 (current: 0.227) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.091 (current: 0.761) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.229 (current: 0.391) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.323 (current: 0.261) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.761 (current: 0.137) (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.712 (current: 0.26) (1 instances)
- DiabetesPedigreeFunction target value -0.045 outside realistic range [0.08, 2.5] (1 instances)
- DiabetesPedigreeFunction already satisfies <= 0.846 (current: 0.627) (1 instances)
- DiabetesPedigreeFunction already satisfies > 0.134 (current: 0.867) (1 instances)

### Glucose
- Glucose already satisfies <= 124.855 (current: 106.0) (1 instances)
- Glucose already satisfies <= 132.272 (current: 103.0) (1 instances)
- Glucose already satisfies > 149.413 (current: 179.0) (1 instances)
- Glucose already satisfies > 96.202 (current: 112.0) (1 instances)
- Glucose already satisfies > 124.623 (current: 150.0) (1 instances)
- Glucose already satisfies <= 157.374 (current: 122.0) (1 instances)
- Glucose already satisfies <= 137.195 (current: 88.0) (1 instances)
- Glucose already satisfies <= 145.66 (current: 112.0) (1 instances)
- Glucose already satisfies > 125.311 (current: 183.0) (1 instances)
- Glucose already satisfies > 119.187 (current: 136.0) (1 instances)
- Glucose already satisfies <= 131.991 (current: 129.0) (1 instances)
- Glucose already satisfies <= 125.226 (current: 44.0) (1 instances)
- Glucose already satisfies > 101.785 (current: 125.0) (1 instances)
- Glucose already satisfies <= 133.173 (current: 85.0) (1 instances)
- Glucose already satisfies <= 132.459 (current: 122.0) (1 instances)
- Glucose already satisfies <= 164.749 (current: 44.0) (1 instances)
- Glucose already satisfies <= 134.457 (current: 106.0) (1 instances)
- Glucose already satisfies > 133.774 (current: 179.0) (1 instances)
- Glucose already satisfies <= 137.766 (current: 100.0) (1 instances)
- Glucose already satisfies <= 133.845 (current: 119.0) (1 instances)
- Glucose already satisfies <= 134.806 (current: 134.0) (1 instances)
- Glucose already satisfies <= 124.989 (current: 112.0) (1 instances)
- Glucose already satisfies <= 128.045 (current: 97.0) (1 instances)
- Glucose already satisfies <= 158.03 (current: 119.0) (1 instances)
- Glucose already satisfies <= 131.794 (current: 118.0) (1 instances)
- Glucose already satisfies <= 130.005 (current: 81.0) (1 instances)
- Glucose already satisfies <= 156.257 (current: 97.0) (1 instances)
- Glucose already satisfies > 104.611 (current: 153.0) (1 instances)
- Glucose already satisfies > 112.95 (current: 183.0) (1 instances)
- Glucose already satisfies > 78.445 (current: 110.0) (1 instances)

### Insulin
- Insulin target value -100.284 outside realistic range [14, 900] (1 instances)
- Insulin already satisfies > 149.695 (current: 478.0) (1 instances)
- Insulin already satisfies <= 153.537 (current: 0.0) (1 instances)
- Insulin target value -18.138 outside realistic range [14, 900] (1 instances)
- Insulin target value -103.759 outside realistic range [14, 900] (1 instances)

### Pregnancies
- Pregnancies already satisfies > 0.452 (current: 2.0) (1 instances)
- Pregnancies already satisfies <= 6.749 (current: 2.0) (1 instances)
- Pregnancies target value -0.164 outside realistic range [0, 20] (1 instances)
- Pregnancies already satisfies <= 7.119 (current: 5.0) (1 instances)
- Pregnancies already satisfies > 5.244 (current: 6.0) (1 instances)
- Pregnancies already satisfies <= 7.734 (current: 4.0) (1 instances)
- Pregnancies already satisfies > 0.579 (current: 6.0) (1 instances)
- Pregnancies already satisfies <= 6.143 (current: 3.0) (1 instances)

### SkinThickness
- SkinThickness already satisfies <= 20.504 (current: 0.0) (1 instances)
- SkinThickness target value 4.099 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies <= 26.252 (current: 25.0) (1 instances)
- SkinThickness already satisfies <= 32.196 (current: 24.0) (1 instances)
- SkinThickness already satisfies <= 37.122 (current: 0.0) (1 instances)
- SkinThickness already satisfies <= 16.425 (current: 0.0) (1 instances)
- SkinThickness target value 2.038 outside realistic range [7, 100] (1 instances)
- SkinThickness already satisfies > 26.688 (current: 29.0) (1 instances)
- SkinThickness already satisfies <= 16.479 (current: 0.0) (1 instances)

## Condition-Level Analysis
- **Total conditions analyzed**: 162
- **Feasible conditions**: 59
- **Condition feasibility rate**: 36.4%

