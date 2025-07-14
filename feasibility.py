import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Any
import argparse
import json

class AutomatedFeasibilityAnalyzer:
    def __init__(self):
        # Define feature constraints for each dataset - ONLY HARDCODED PART
        self.feature_constraints = {
            'heart': {
                'immutable': ['age', 'sex'],
                'partially_immutable': {
                    'ca': (0, 4),     # Number of major vessels (0-4)
                    'thal': (0, 3),   # Thalassemia type (0-3)
                    'cp': (0, 3),     # Chest pain type (0-3)
                    'slope': (0, 2),  # Slope of peak exercise ST segment (0-2)
                    'restecg': (0, 2) # Resting ECG results (0-2)
                },
                'mutable': ['trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak'],
                'realistic_ranges': {
                    'age': (18, 100),         # Age in years
                    'trestbps': (80, 220),    # Resting blood pressure (80-220 mmHg)
                    'chol': (100, 600),       # Serum cholesterol (100-600 mg/dl)
                    'thalach': (60, 220),     # Maximum heart rate (60-220 bpm)
                    'oldpeak': (0, 6.2),      # ST depression (0-6.2)
                    'fbs': (0, 1),            # Fasting blood sugar (0-1)
                    'exang': (0, 1),          # Exercise induced angina (0-1)
                    'sex': (0, 1),            # Sex (0-1)
                    'ca': (0, 4),             # Number of major vessels (0-4)
                    'thal': (0, 3),           # Thalassemia (0-3)
                    'cp': (0, 3),             # Chest pain type (0-3)
                    'restecg': (0, 2),        # Resting ECG (0-2)
                    'slope': (0, 2)           # Slope (0-2)
                }
            },
            'diabetes': {
                'immutable': ['Age'],
                'partially_immutable': {
                    'Pregnancies': (0, 20),
                    'Age': (21, 100)  # Can't change significantly
                },
                'mutable': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'],
                'realistic_ranges': {
                    'Pregnancies': (0, 20),
                    'Glucose': (40, 300),           # Blood glucose (40-300 mg/dL)
                    'BloodPressure': (40, 200),     # Diastolic blood pressure (40-200 mmHg)
                    'SkinThickness': (7, 100),      # Triceps skin fold thickness (7-100 mm)
                    'Insulin': (14, 900),           # 2-Hour serum insulin (14-900 mu U/ml)
                    'BMI': (15, 60),                # Body mass index (15-60)
                    'DiabetesPedigreeFunction': (0.08, 2.5),  # Diabetes pedigree function
                    'Age': (21, 100)                # Age in years
                }
            },
            'adult': {
                'immutable': ['age', 'sex', 'race'],
                'partially_immutable': {
                    'age': (17, 90),
                    'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 
                                'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
                },
                'mutable': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
                           'capital-gain', 'capital-loss', 'hours-per-week'],
                'realistic_ranges': {
                    'age': (17, 90),
                    'capital-gain': (0, 100000),
                    'capital-loss': (0, 5000),
                    'hours-per-week': (1, 99)
                }
            },
            'bank': {
                'immutable': ['age'],
                'partially_immutable': {
                    'age': (18, 100),
                    'dependents': (0, 20)
                },
                'mutable': ['revolving', 'nbr_30_59_days_past_due_not_worse', 'debt_ratio', 'monthly_income',
                           'nbr_open_credits_and_loans', 'nbr_90_days_late', 'nbr_real_estate_loans_or_lines',
                           'nbr_60_89_days_past_due_not_worse'],
                'realistic_ranges': {
                    'age': (18, 100),
                    'revolving': (0, 200000),
                    'debt_ratio': (0, 50),
                    'monthly_income': (0, 100000),
                    'nbr_open_credits_and_loans': (0, 50),
                    'nbr_90_days_late': (0, 50),
                    'nbr_real_estate_loans_or_lines': (0, 20),
                    'nbr_30_59_days_past_due_not_worse': (0, 50),
                    'nbr_60_89_days_past_due_not_worse': (0, 50),
                    'dependents': (0, 20)
                }
            },
            'german': {
                'immutable': ['age', 'personal_status_sex', 'foreign_worker'],
                'partially_immutable': {
                    'age': (18, 100),
                    'duration_in_month': (4, 72),
                    'credits_this_bank': (1, 10),
                    'people_under_maintenance': (0, 10)
                },
                'mutable': ['account_check_status', 'credit_history', 'purpose', 'credit_amount', 'savings',
                           'present_emp_since', 'installment_as_income_perc', 'other_debtors', 'present_res_since',
                           'property', 'other_installment_plans', 'housing', 'job', 'telephone'],
                'realistic_ranges': {
                    'age': (18, 100),
                    'duration_in_month': (4, 72),
                    'credit_amount': (250, 20000),
                    'installment_as_income_perc': (1, 4),
                    'present_res_since': (1, 4),
                    'credits_this_bank': (1, 10),
                    'people_under_maintenance': (0, 10)
                }
            }
        }
    
    def parse_counterfactual_rules(self, rules_str: str) -> List[Tuple[str, str, float]]:
        """Parse counterfactual rules string into individual conditions"""
        conditions = []
        if pd.isna(rules_str) or rules_str == '' or rules_str == 'nan':
            return conditions
        
        # Split by 'and' and clean up
        parts = rules_str.split(' and ')
        
        for part in parts:
            part = part.strip()
            # Match patterns like "feature <= value" or "feature > value"
            match = re.match(r'(\w+)\s*([<>=!]+)\s*(-?\d+\.?\d*)', part)
            if match:
                feature, operator, value = match.groups()
                conditions.append((feature, operator, float(value)))
        
        return conditions
    
    def check_feature_feasibility(self, dataset: str, feature: str, operator: str, 
                                target_value: float, original_value: float) -> Dict[str, Any]:
        """Check if a feature change is feasible"""
        constraints = self.feature_constraints.get(dataset, {})
        
        result = {
            'feature': feature,
            'operator': operator,
            'target_value': target_value,
            'original_value': original_value,
            'feasible': True,
            'reason': '',
            'constraint_type': 'mutable',
            'change_required': abs(target_value - original_value) if original_value is not None else None
        }
        
        # Check if feature is immutable
        if feature in constraints.get('immutable', []):
            result['feasible'] = False
            result['reason'] = f"{feature} is immutable and cannot be changed"
            result['constraint_type'] = 'immutable'
            return result
        
        # Check if feature is partially immutable
        if feature in constraints.get('partially_immutable', {}):
            constraint = constraints['partially_immutable'][feature]
            result['constraint_type'] = 'partially_immutable'
            
            if isinstance(constraint, tuple):  # Range constraint
                min_val, max_val = constraint
                if target_value < min_val or target_value > max_val:
                    result['feasible'] = False
                    result['reason'] = f"{feature} target value {target_value} outside realistic range [{min_val}, {max_val}]"
                    return result
        
        # Check realistic ranges for all features
        if feature in constraints.get('realistic_ranges', {}):
            min_val, max_val = constraints['realistic_ranges'][feature]
            if target_value < min_val or target_value > max_val:
                result['feasible'] = False
                result['reason'] = f"{feature} target value {target_value} outside realistic range [{min_val}, {max_val}]"
                return result
        
        # Check if the change makes logical sense given the operator
        if original_value is not None:
            if operator == '<=':
                if original_value <= target_value:
                    result['feasible'] = False
                    result['reason'] = f"{feature} already satisfies <= {target_value} (current: {original_value})"
            elif operator == '>=':
                if original_value >= target_value:
                    result['feasible'] = False
                    result['reason'] = f"{feature} already satisfies >= {target_value} (current: {original_value})"
            elif operator == '>':
                if original_value > target_value:
                    result['feasible'] = False
                    result['reason'] = f"{feature} already satisfies > {target_value} (current: {original_value})"
            elif operator == '<':
                if original_value < target_value:
                    result['feasible'] = False
                    result['reason'] = f"{feature} already satisfies < {target_value} (current: {original_value})"
        
        return result
    
    def analyze_instance_feasibility(self, instance_id: Any, counterfactual_rules: str, 
                                   original_data: pd.DataFrame, dataset: str) -> Dict[str, Any]:
        """Analyze feasibility of counterfactual rules for a specific instance"""
        
        # Get original instance data
        original_instance = original_data[original_data.index == instance_id]
        if original_instance.empty:
            return {
                'instance_id': instance_id,
                'feasible': False,
                'reason': f"Instance {instance_id} not found in original data",
                'feature_analysis': [],
                'total_conditions': 0,
                'feasible_conditions': 0
            }
        
        original_values = original_instance.iloc[0].to_dict()
        
        # Parse counterfactual rules
        conditions = self.parse_counterfactual_rules(counterfactual_rules)
        
        feature_analysis = []
        overall_feasible = True
        
        for feature, operator, target_value in conditions:
            if feature in original_values:
                original_value = original_values[feature]
                feasibility = self.check_feature_feasibility(
                    dataset, feature, operator, target_value, original_value
                )
                feature_analysis.append(feasibility)
                
                if not feasibility['feasible']:
                    overall_feasible = False
            else:
                feature_analysis.append({
                    'feature': feature,
                    'operator': operator,
                    'target_value': target_value,
                    'original_value': None,
                    'feasible': False,
                    'reason': f"Feature {feature} not found in original data",
                    'constraint_type': 'unknown',
                    'change_required': None
                })
                overall_feasible = False
        
        return {
            'instance_id': instance_id,
            'counterfactual_rules': counterfactual_rules,
            'feasible': overall_feasible,
            'feature_analysis': feature_analysis,
            'total_conditions': len(conditions),
            'feasible_conditions': sum(1 for f in feature_analysis if f['feasible'])
        }

def load_and_analyze_feasibility(counterfactual_csv_path: str, validation_csv_path: str, 
                               original_data_path: str, dataset_name: str):
    """Load CSV files and analyze counterfactual feasibility"""
    
    # Initialize analyzer
    analyzer = AutomatedFeasibilityAnalyzer()
    
    # Load data
    try:
        print(f"Loading data for {dataset_name}...")
        cf_df = pd.read_csv(counterfactual_csv_path)
        val_df = pd.read_csv(validation_csv_path)
        original_data = pd.read_csv(original_data_path)
        
        print(f"  - Counterfactual data: {len(cf_df)} rows")
        print(f"  - Validation data: {len(val_df)} rows")
        print(f"  - Original data: {len(original_data)} rows")
        
        # Handle different index column names
        instance_col = None
        if 'Instance_ID' in cf_df.columns:
            instance_col = 'Instance_ID'
        elif 'instance_id' in cf_df.columns:
            instance_col = 'instance_id'
        elif 'Instance' in cf_df.columns:
            instance_col = 'Instance'
        
        if instance_col:
            cf_df = cf_df.set_index(instance_col)
        
        # Handle validation data index
        val_instance_col = None
        if 'instance' in val_df.columns:
            val_instance_col = 'instance'
        elif 'Instance_ID' in val_df.columns:
            val_instance_col = 'Instance_ID'
        elif 'instance_id' in val_df.columns:
            val_instance_col = 'instance_id'
        
        if val_instance_col:
            val_df = val_df.set_index(val_instance_col)
        
        # Set original data index if there's an ID column
        if len(original_data.columns) > 0 and original_data.index.name is None:
            # Try to find an ID column or use row number
            if 'id' in original_data.columns:
                original_data = original_data.set_index('id')
            elif 'ID' in original_data.columns:
                original_data = original_data.set_index('ID')
            else:
                # Use row numbers as index
                original_data.index = range(len(original_data))
                
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Filter for successful counterfactuals only (✓)
    successful_mask = val_df['result'] == '✓'
    successful_instances = val_df[successful_mask].index.tolist()
    
    print(f"\nAnalyzing {dataset_name.upper()} dataset:")
    print(f"  Total counterfactuals: {len(cf_df)}")
    print(f"  Successful counterfactuals: {len(successful_instances)}")
    print(f"  Success rate: {len(successful_instances)/len(cf_df)*100:.1f}%")
    
    if len(successful_instances) == 0:
        print("No successful counterfactuals to analyze!")
        return []
    
    results = []
    
    for instance_id in successful_instances:
        if instance_id in cf_df.index:
            row = cf_df.loc[instance_id]
            
            # Handle different column names for rules
            rules_col = None
            if 'Counterfactual_Rules' in row.index:
                rules_col = 'Counterfactual_Rules'
            elif 'counterfactual_rules' in row.index:
                rules_col = 'counterfactual_rules'
            elif 'rules' in row.index:
                rules_col = 'rules'
            
            if rules_col is None:
                print(f"Warning: Could not find rules column for instance {instance_id}")
                continue
            
            rules = row[rules_col]
            
            result = analyzer.analyze_instance_feasibility(
                instance_id, rules, original_data, dataset_name
            )
            results.append(result)
    
    return results

def generate_feasibility_report(results: List[Dict], dataset_name: str) -> str:
    """Generate a comprehensive feasibility report"""
    
    if not results:
        return "No results to analyze."
    
    report = f"# Counterfactual Feasibility Analysis Report - {dataset_name.upper()}\n\n"
    
    # Overall statistics
    total_analyzed = len(results)
    feasible_count = sum(1 for r in results if r['feasible'])
    feasibility_rate = feasible_count / total_analyzed * 100 if total_analyzed > 0 else 0
    
    report += f"## Executive Summary\n"
    report += f"- **Total Successful Counterfactuals Analyzed**: {total_analyzed}\n"
    report += f"- **Feasible Counterfactuals**: {feasible_count}\n"
    report += f"- **Feasibility Rate**: {feasibility_rate:.1f}%\n\n"
    
    # Feature constraint analysis
    constraint_counts = {'immutable': 0, 'partially_immutable': 0, 'mutable': 0, 'unknown': 0}
    feature_involvement = {}
    infeasible_features = {}
    change_magnitudes = {}
    
    for result in results:
        for feature_result in result['feature_analysis']:
            feature = feature_result['feature']
            constraint_type = feature_result['constraint_type']
            
            constraint_counts[constraint_type] += 1
            
            if feature not in feature_involvement:
                feature_involvement[feature] = {'total': 0, 'feasible': 0}
            
            feature_involvement[feature]['total'] += 1
            if feature_result['feasible']:
                feature_involvement[feature]['feasible'] += 1
                
                # Track change magnitudes for feasible changes
                if feature_result['change_required'] is not None:
                    if feature not in change_magnitudes:
                        change_magnitudes[feature] = []
                    change_magnitudes[feature].append(feature_result['change_required'])
            else:
                if feature not in infeasible_features:
                    infeasible_features[feature] = []
                infeasible_features[feature].append(feature_result['reason'])
    
    report += f"## Feature Constraint Analysis\n"
    report += f"- **Immutable features involved**: {constraint_counts['immutable']}\n"
    report += f"- **Partially immutable features involved**: {constraint_counts['partially_immutable']}\n"
    report += f"- **Mutable features involved**: {constraint_counts['mutable']}\n"
    report += f"- **Unknown features involved**: {constraint_counts['unknown']}\n\n"
    
    # Feature-wise feasibility
    report += f"## Feature-wise Feasibility\n"
    for feature, counts in sorted(feature_involvement.items()):
        feasible_rate = counts['feasible'] / counts['total'] * 100
        report += f"- **{feature}**: {counts['feasible']}/{counts['total']} ({feasible_rate:.1f}% feasible)\n"
        
        # Add change magnitude statistics
        if feature in change_magnitudes and change_magnitudes[feature]:
            changes = change_magnitudes[feature]
            avg_change = np.mean(changes)
            max_change = np.max(changes)
            report += f"  - Average change magnitude: {avg_change:.3f}\n"
            report += f"  - Maximum change magnitude: {max_change:.3f}\n"
    
    # Common issues
    if infeasible_features:
        report += f"\n## Common Feasibility Issues\n"
        for feature, issues in sorted(infeasible_features.items()):
            report += f"\n### {feature}\n"
            unique_issues = list(set(issues))
            for issue in unique_issues:
                count = issues.count(issue)
                report += f"- {issue} ({count} instances)\n"
    
    # Summary statistics
    total_conditions = sum(r['total_conditions'] for r in results)
    total_feasible_conditions = sum(r['feasible_conditions'] for r in results)
    condition_feasibility_rate = total_feasible_conditions / total_conditions * 100 if total_conditions > 0 else 0
    
    report += f"\n## Condition-Level Analysis\n"
    report += f"- **Total conditions analyzed**: {total_conditions}\n"
    report += f"- **Feasible conditions**: {total_feasible_conditions}\n"
    report += f"- **Condition feasibility rate**: {condition_feasibility_rate:.1f}%\n\n"
    
    return report

def batch_analyze_datasets(base_dir: str = ".", output_dir: str = "feasibility_reports"):
    """Analyze feasibility for all datasets in a directory"""
    
    datasets = ['heart', 'diabetes', 'adult', 'bank', 'german']
    models = ['mlp', 'decision_tree', 'logistic_regression', 'random_forest', 'xgboost']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    overall_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset.upper()} DATASET")
        print('='*60)
        
        dataset_results = {}
        
        for model in models:
            print(f"\n--- {model.upper()} Model ---")
            
            # Updated file paths
            cf_file = os.path.join(base_dir, "Evaluation", "results", f"{dataset}_{model}_counterfactual_report.csv")
            val_file = os.path.join(base_dir, "validation_results", f"{dataset}_{model}_validation_summary.csv")
            orig_file = os.path.join(base_dir, "Evaluation", "data", f"{dataset}.csv")
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [cf_file, val_file, orig_file]):
                missing_files = [f for f in [cf_file, val_file, orig_file] if not os.path.exists(f)]
                print(f"Missing files for {dataset}_{model}: {missing_files}")
                continue
            
            try:
                # Analyze feasibility
                results = load_and_analyze_feasibility(cf_file, val_file, orig_file, dataset)
                
                if results:
                    # Generate report
                    report = generate_feasibility_report(results, f"{dataset}_{model}")
                    
                    # Save report
                    report_file = os.path.join(output_dir, f"{dataset}_{model}_feasibility_report.md")
                    with open(report_file, 'w') as f:
                        f.write(report)
                    
                    # Store results
                    dataset_results[model] = {
                        'total_analyzed': len(results),
                        'feasible_count': sum(1 for r in results if r['feasible']),
                        'feasibility_rate': sum(1 for r in results if r['feasible']) / len(results) * 100 if results else 0
                    }
                    
                    print(f"  Analysis complete: {dataset_results[model]['feasible_count']}/{dataset_results[model]['total_analyzed']} feasible ({dataset_results[model]['feasibility_rate']:.1f}%)")
                    print(f"  Report saved: {report_file}")
                
            except Exception as e:
                print(f"  Error analyzing {dataset}_{model}: {e}")
        
        overall_results[dataset] = dataset_results
    
    # Generate summary report
    summary_report = generate_summary_report(overall_results)
    summary_file = os.path.join(output_dir, "feasibility_summary.md")
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print('='*60)
    print(f"Summary report saved: {summary_file}")
    
    return overall_results

def generate_summary_report(overall_results: Dict) -> str:
    """Generate a summary report across all datasets and models"""
    
    report = "# Counterfactual Feasibility Analysis - Summary Report\n\n"
    
    # Overall statistics
    total_datasets = len(overall_results)
    total_models = sum(len(dataset_results) for dataset_results in overall_results.values())
    
    report += f"## Overview\n"
    report += f"- **Datasets analyzed**: {total_datasets}\n"
    report += f"- **Dataset-model combinations**: {total_models}\n\n"
    
    # Dataset-wise summary
    report += f"## Dataset-wise Results\n\n"
    
    for dataset, dataset_results in overall_results.items():
        if not dataset_results:
            continue
            
        report += f"### {dataset.upper()}\n\n"
        report += f"| Model | Total Analyzed | Feasible | Feasibility Rate |\n"
        report += f"|-------|---------------|----------|------------------|\n"
        
        dataset_totals = {'analyzed': 0, 'feasible': 0}
        
        for model, results in dataset_results.items():
            report += f"| {model} | {results['total_analyzed']} | {results['feasible_count']} | {results['feasibility_rate']:.1f}% |\n"
            dataset_totals['analyzed'] += results['total_analyzed']
            dataset_totals['feasible'] += results['feasible_count']
        
        if dataset_totals['analyzed'] > 0:
            dataset_feasibility = dataset_totals['feasible'] / dataset_totals['analyzed'] * 100
            report += f"| **Total** | **{dataset_totals['analyzed']}** | **{dataset_totals['feasible']}** | **{dataset_feasibility:.1f}%** |\n"
        
        report += "\n"
    
    # Model-wise summary
    report += f"## Model-wise Results\n\n"
    
    model_totals = {}
    for dataset_results in overall_results.values():
        for model, results in dataset_results.items():
            if model not in model_totals:
                model_totals[model] = {'analyzed': 0, 'feasible': 0}
            model_totals[model]['analyzed'] += results['total_analyzed']
            model_totals[model]['feasible'] += results['feasible_count']
    
    report += f"| Model | Total Analyzed | Feasible | Feasibility Rate |\n"
    report += f"|-------|---------------|----------|------------------|\n"
    
    for model, totals in model_totals.items():
        if totals['analyzed'] > 0:
            feasibility_rate = totals['feasible'] / totals['analyzed'] * 100
            report += f"| {model} | {totals['analyzed']} | {totals['feasible']} | {feasibility_rate:.1f}% |\n"
    
    return report

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Analyze counterfactual feasibility")
    parser.add_argument("--base-dir", default=".", help="Base directory containing the project structure")
    parser.add_argument("--output-dir", default="feasibility_reports", help="Output directory for reports")
    parser.add_argument("--dataset", help="Specific dataset to analyze (optional)")
    parser.add_argument("--model", help="Specific model to analyze (optional)")
    
    args = parser.parse_args()
    
    if args.dataset and args.model:
        # Analyze specific dataset-model combination
        cf_file = os.path.join(args.base_dir, "Evaluation", "results", f"{args.dataset}_{args.model}_counterfactual_report.csv")
        val_file = os.path.join(args.base_dir, "validation_results", f"{args.dataset}_{args.model}_validation_summary.csv")
        orig_file = os.path.join(args.base_dir, "Evaluation", "data", f"{args.dataset}.csv")
        
        results = load_and_analyze_feasibility(cf_file, val_file, orig_file, args.dataset)
        
        if results:
            report = generate_feasibility_report(results, f"{args.dataset}_{args.model}")
            print(report)
    else:
        # Batch analyze all datasets
        batch_analyze_datasets(args.base_dir, args.output_dir)

if __name__ == "__main__":
    main()