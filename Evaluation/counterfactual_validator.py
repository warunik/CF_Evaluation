import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Import the config
from config import DATASETS, ML_MODELS

class CounterfactualValidator:
    def __init__(self, dataset_name=None, model_name=None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dataset_config = None
        self.data = None
        self.model = None
        self.feature_frequency = {}  # Track feature usage frequency
        self.feature_stats = {}  # Store feature statistics for adaptive scaling
        
    def parse_filename(self, filename):
        """Parse filename to extract dataset and model name"""
        # Extract basename without extension
        basename = os.path.splitext(os.path.basename(filename))[0]
        
        # Expected pattern: {dataset_name}_{ml_model}_counterfactual_report
        # Remove the '_counterfactual_report' suffix
        if basename.endswith('_counterfactual_report'):
            basename = basename[:-22]  # Remove '_counterfactual_report'
        
        # Split by underscore and try to match dataset and model
        parts = basename.split('_')
        
        detected_dataset = None
        detected_model = None
        
        # Try to find dataset name
        for dataset_key in DATASETS.keys():
            if dataset_key in parts:
                detected_dataset = dataset_key
                break
        
        # Try to find model name
        for model_key in ML_MODELS.keys():
            if model_key in parts:
                detected_model = model_key
                break
        
        # If standard parsing fails, try more flexible matching
        if not detected_dataset or not detected_model:
            print(f"Standard parsing failed for {basename}")
            print("Trying flexible matching...")
            
            # Try to match any combination
            for dataset_key in DATASETS.keys():
                for model_key in ML_MODELS.keys():
                    expected_pattern = f"{dataset_key}_{model_key}"
                    if expected_pattern in basename:
                        detected_dataset = dataset_key
                        detected_model = model_key
                        break
                if detected_dataset and detected_model:
                    break
        
        return detected_dataset, detected_model
    
    def setup_from_filename(self, counterfactual_csv_path):
        """Setup validator based on the counterfactual report filename"""
        print(f"Analyzing filename: {counterfactual_csv_path}")
        
        dataset_name, model_name = self.parse_filename(counterfactual_csv_path)
        
        if not dataset_name or not model_name:
            print(f"Could not parse dataset and model from filename: {counterfactual_csv_path}")
            print("Available datasets:", list(DATASETS.keys()))
            print("Available models:", list(ML_MODELS.keys()))
            return False
        
        print(f"Detected dataset: {dataset_name}")
        print(f"Detected model: {model_name}")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dataset_config = DATASETS[dataset_name]
        
        return True
    
    def calculate_feature_stats(self):
        """Calculate feature statistics for adaptive scaling"""
        print("Calculating feature statistics for adaptive scaling...")
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in self.data.columns if col != self.dataset_config["target_column"]]
        
        for feature in feature_cols:
            if feature in self.data.columns:
                values = self.data[feature].dropna()
                
                # Calculate statistics
                stats = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'iqr': values.quantile(0.75) - values.quantile(0.25),
                    'range': values.max() - values.min()
                }
                
                # Calculate adaptive step size
                # Use 5% of standard deviation as base step, with minimum and maximum bounds
                base_step = stats['std'] * 0.05
                min_step = stats['iqr'] * 0.01  # 1% of IQR as minimum
                max_step = stats['iqr'] * 0.1   # 10% of IQR as maximum
                
                # Ensure step is within reasonable bounds
                if base_step < min_step:
                    adaptive_step = min_step
                elif base_step > max_step:
                    adaptive_step = max_step
                else:
                    adaptive_step = base_step
                
                # Handle edge cases
                if adaptive_step == 0 or np.isnan(adaptive_step):
                    adaptive_step = abs(stats['range']) * 0.01  # 1% of range as fallback
                
                # Final fallback for constant features
                if adaptive_step == 0:
                    adaptive_step = 0.1
                
                stats['adaptive_step'] = adaptive_step
                self.feature_stats[feature] = stats
                
                print(f"Feature '{feature}': range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                      f"std={stats['std']:.3f}, adaptive_step={adaptive_step:.3f}")
    
    def load_data(self):
        """Load the dataset"""
        try:
            self.data = pd.read_csv(self.dataset_config["path"])
            print(f"Dataset loaded successfully: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Calculate feature statistics for adaptive scaling
            self.calculate_feature_stats()
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def load_model(self):
        """Load the specific pretrained model"""
        if self.model_name not in self.dataset_config["model_paths"]:
            print(f"Model {self.model_name} not found in dataset config")
            return False
        
        model_path = self.dataset_config["model_paths"][self.model_name]
        
        try:
            # Try different loading methods
            try:
                self.model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            print(f"✓ {self.model_name} model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            return False
    
    def get_original_sample(self, instance_id):
        """Get original sample by instance ID"""
        if instance_id >= len(self.data):
            print(f"Warning: Instance ID {instance_id} out of range")
            return None
        
        return self.data.iloc[instance_id].copy()
    
    def parse_counterfactual_rules(self, rules_str):
        """Parse counterfactual rules string into conditions"""
        conditions = []
        
        # Handle NaN or None values
        if pd.isna(rules_str) or rules_str is None:
            print("Warning: No counterfactual rules provided (NaN or None)")
            return conditions
        
        # Convert to string if it's not already
        rules_str = str(rules_str)
        
        # Split by 'and' to get individual conditions
        rule_parts = rules_str.split(' and ')
        
        for part in rule_parts:
            part = part.strip()
            
            # Handle different operators
            if ' <= ' in part:
                feature, value = part.split(' <= ')
                conditions.append((feature.strip(), '<=', float(value.strip())))
            elif ' >= ' in part:
                feature, value = part.split(' >= ')
                conditions.append((feature.strip(), '>=', float(value.strip())))
            elif ' > ' in part:
                feature, value = part.split(' > ')
                conditions.append((feature.strip(), '>', float(value.strip())))
            elif ' < ' in part:
                feature, value = part.split(' < ')
                conditions.append((feature.strip(), '<', float(value.strip())))
            elif ' = ' in part:
                feature, value = part.split(' = ')
                conditions.append((feature.strip(), '=', float(value.strip())))
        
        return conditions
    
    def track_feature_usage(self, conditions):
        """Track which features are being used in counterfactuals"""
        for feature, operator, value in conditions:
            if feature not in self.feature_frequency:
                self.feature_frequency[feature] = 0
            self.feature_frequency[feature] += 1
    
    def apply_counterfactual(self, original_sample, conditions):
        """Apply counterfactual conditions to create modified sample with adaptive scaling"""
        modified_sample = original_sample.copy()
        
        print(f"Applying counterfactual conditions:")
        
        for feature, operator, value in conditions:
            if feature in modified_sample.index:
                original_value = modified_sample[feature]
                
                # Get adaptive step size for this feature
                step_size = self.feature_stats.get(feature, {}).get('adaptive_step', 0.1)
                
                print(f"  {feature}: {original_value} -> ", end="")
                
                if operator == '<=':
                    # Set to a value that satisfies the condition (threshold - step_size)
                    new_value = value - step_size
                elif operator == '>=':
                    # Set to a value that satisfies the condition (threshold + step_size)
                    new_value = value + step_size
                elif operator == '>':
                    # Set to a value that satisfies the condition (threshold + step_size)
                    new_value = value + step_size
                elif operator == '<':
                    # Set to a value that satisfies the condition (threshold - step_size)
                    new_value = value - step_size
                elif operator == '=':
                    new_value = value
                
                # Ensure the new value is within reasonable bounds
                feature_min = self.feature_stats.get(feature, {}).get('min', float('-inf'))
                feature_max = self.feature_stats.get(feature, {}).get('max', float('inf'))
                new_value = max(feature_min, min(feature_max, new_value))
                
                modified_sample[feature] = new_value
                
                print(f"{new_value:.3f} (step: {step_size:.3f}, condition: {feature} {operator} {value})")
                
            else:
                print(f"Warning: Feature '{feature}' not found in sample")
        
        return modified_sample
    
    def predict_with_model(self, sample_data):
        """Make prediction using the loaded model"""
        # Prepare feature data (exclude target if present)
        if isinstance(sample_data, pd.Series):
            feature_cols = [col for col in sample_data.index if col != self.dataset_config["target_column"]]
            features = sample_data[feature_cols].values.reshape(1, -1)
        else:
            # If it's already a DataFrame or array
            features = sample_data.reshape(1, -1) if len(sample_data.shape) == 1 else sample_data
        
        try:
            prediction = self.model.predict(features)[0]
            prob = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
            return {
                'prediction': prediction,
                'probability': prob,
                'confidence': max(prob) if prob is not None else None
            }
        except Exception as e:
            print(f"✗ Error predicting with {self.model_name}: {e}")
            # Try alternative approach
            try:
                if isinstance(sample_data, pd.Series):
                    feature_cols = [col for col in sample_data.index if col != self.dataset_config["target_column"]]
                    features_df = pd.DataFrame([sample_data[feature_cols].values], columns=feature_cols)
                    prediction = self.model.predict(features_df)[0]
                    prob = self.model.predict_proba(features_df)[0] if hasattr(self.model, 'predict_proba') else None
                    return {
                        'prediction': prediction,
                        'probability': prob,
                        'confidence': max(prob) if prob is not None else None
                    }
                else:
                    # Try using feature names
                    if hasattr(self.model, 'feature_names_in_'):
                        feature_names = self.model.feature_names_in_
                    else:
                        feature_names = [col for col in self.data.columns if col != self.dataset_config["target_column"]]
                    
                    if len(features.flatten()) == len(feature_names):
                        features_df = pd.DataFrame(features, columns=feature_names)
                        prediction = self.model.predict(features_df)[0]
                        prob = self.model.predict_proba(features_df)[0] if hasattr(self.model, 'predict_proba') else None
                        return {
                            'prediction': prediction,
                            'probability': prob,
                            'confidence': max(prob) if prob is not None else None
                        }
            except Exception as e2:
                print(f"✗ Alternative approach also failed: {e2}")
                return None
    
    def validate_counterfactual_sample(self, instance_id, actual_class, predicted_class, 
                                     contrast_class, counterfactual_rules, prediction_confidence):
        """Validate a single counterfactual sample"""
        
        print(f"\nValidating Instance {instance_id}")
        
        # Get original sample
        original_sample = self.get_original_sample(instance_id)
        if original_sample is None:
            return None
        
        # Parse counterfactual rules
        conditions = self.parse_counterfactual_rules(counterfactual_rules)
        
        # Track feature usage
        if conditions:
            self.track_feature_usage(conditions)
        
        # If no conditions (NaN rules), skip this sample
        if not conditions:
            print("Skipping validation due to missing counterfactual rules")
            return {
                'instance_id': instance_id,
                'skipped': True,
                'reason': 'No counterfactual rules provided',
                'original_class': actual_class,
                'expected_contrast_class': contrast_class,
                'original_prediction_confidence': prediction_confidence
            }
        
        # Apply counterfactual
        modified_sample = self.apply_counterfactual(original_sample, conditions)
        
        # Predict with model
        original_prediction = self.predict_with_model(original_sample)
        modified_prediction = self.predict_with_model(modified_sample)
        
        # Analyze results
        results = {
            'instance_id': instance_id,
            'original_sample': original_sample,
            'modified_sample': modified_sample,
            'conditions': conditions,
            'original_prediction': original_prediction,
            'modified_prediction': modified_prediction,
            'original_class': actual_class,
            'expected_contrast_class': contrast_class,
            'original_prediction_confidence': prediction_confidence,
            'skipped': False
        }
        
        if original_prediction is not None and modified_prediction is not None:
            orig_pred = original_prediction['prediction']
            mod_pred = modified_prediction['prediction']
            
            # Map predictions to class labels
            orig_class = self.dataset_config["class_labels"][orig_pred]
            mod_class = self.dataset_config["class_labels"][mod_pred]
            
            # Check if counterfactual worked
            success = mod_class == contrast_class
            
            results['validation_result'] = {
                'original_prediction': orig_pred,
                'modified_prediction': mod_pred,
                'original_class': orig_class,
                'modified_class': mod_class,
                'counterfactual_success': success,
                'original_confidence': original_prediction['confidence'],
                'modified_confidence': modified_prediction['confidence']
            }
            
            print(f"  Original prediction: {orig_class} (confidence: {original_prediction['confidence']:.3f})")
            print(f"  Modified prediction: {mod_class} (confidence: {modified_prediction['confidence']:.3f})")
            print(f"  Expected contrast: {contrast_class}")
            print(f"  Counterfactual success: {'✓' if success else '✗'}")
            
        else:
            results['validation_result'] = None
        
        return results
    
    def generate_feature_frequency_report(self, output_path):
        """Generate a report of feature usage frequency"""
        if not self.feature_frequency:
            print("No feature frequency data available. Run validation first.")
            return None
        
        # Sort features by frequency
        sorted_features = sorted(self.feature_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Create DataFrame
        df_features = pd.DataFrame(sorted_features, columns=['Feature', 'Frequency'])
        
        # Add feature statistics
        for idx, row in df_features.iterrows():
            feature = row['Feature']
            if feature in self.feature_stats:
                stats = self.feature_stats[feature]
                df_features.loc[idx, 'Mean'] = stats['mean']
                df_features.loc[idx, 'Std'] = stats['std']
                df_features.loc[idx, 'Adaptive_Step'] = stats['adaptive_step']
                df_features.loc[idx, 'Range'] = stats['range']
        
        # Calculate percentages
        total_usage = sum(self.feature_frequency.values())
        df_features['Percentage'] = (df_features['Frequency'] / total_usage * 100).round(2)
        
        # Add ranking
        df_features['Rank'] = range(1, len(df_features) + 1)
        
        # Reorder columns
        df_features = df_features[['Rank', 'Feature', 'Frequency', 'Percentage', 'Mean', 'Std', 'Adaptive_Step', 'Range']]
        
        # Save to CSV
        df_features.to_csv(output_path, index=False)
        print(f"Feature frequency report saved to: {output_path}")
        
        # Display summary
        print("\nFEATURE USAGE FREQUENCY REPORT")
        print("=" * 50)
        print(f"Dataset: {self.dataset_config['name']}")
        print(f"Model: {ML_MODELS[self.model_name]}")
        print(f"Total counterfactual conditions analyzed: {total_usage}")
        print(f"Unique features used: {len(self.feature_frequency)}")
        print("\nMost frequently used features:")
        print(df_features.head(5).to_string(index=False))
        
        return df_features
    
    def generate_summary_report_csv(self, validation_results, output_path):
        """Generate a summary report in CSV format"""
        
        print("\nGenerating summary report CSV...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Prepare data for CSV
        report_data = []
        
        for result in validation_results:
            if result:
                row = {
                    'instance': result['instance_id'],
                    'original_class': result.get('original_class', 'Unknown'),
                    'expected_contrast': result.get('expected_contrast_class', 'Unknown'),
                    'dataset': self.dataset_name,
                    'model': self.model_name
                }
                
                if result.get('skipped', False):
                    row['result'] = 'SKIPPED'
                    row['reason'] = result.get('reason', 'Unknown')
                else:
                    if result.get('validation_result') is not None:
                        success = result['validation_result']['counterfactual_success']
                        row['result'] = '✓' if success else '✗'
                        row['original_prediction'] = result['validation_result']['original_class']
                        row['modified_prediction'] = result['validation_result']['modified_class']
                        row['original_confidence'] = result['validation_result']['original_confidence']
                        row['modified_confidence'] = result['validation_result']['modified_confidence']
                    else:
                        row['result'] = 'FAILED'
                        row['reason'] = 'Prediction failed'
                
                report_data.append(row)
        
        # Create DataFrame and save
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(output_path, index=False)
        
        print(f"Summary report saved to: {output_path}")
        
        # Display preview
        print("\nPreview of the report:")
        print(df_report.head(10).to_string(index=False))
        
        return df_report
    
    def generate_comprehensive_report(self, validation_results, save_csv=True, output_dir="./"):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        # Dataset and model info
        print(f"Dataset: {self.dataset_config['name']} ({self.dataset_name})")
        print(f"Model: {ML_MODELS[self.model_name]} ({self.model_name})")
        
        # Overall statistics
        total_samples = len(validation_results)
        print(f"Total Samples Validated: {total_samples}")
        
        # Success rates
        print("\nCOUNTERFACTUAL SUCCESS RATES:")
        print("-" * 40)
        
        successes = sum(1 for result in validation_results 
                       if result and not result.get('skipped', False)
                       and result.get('validation_result') is not None
                       and result['validation_result']['counterfactual_success'])
        
        total_valid = sum(1 for result in validation_results 
                         if result and not result.get('skipped', False)
                         and result.get('validation_result') is not None)
        
        skipped = sum(1 for result in validation_results if result and result.get('skipped', False))
        
        success_rate = successes / total_valid * 100 if total_valid > 0 else 0
        
        print(f"Successful: {successes}/{total_valid} ({success_rate:.1f}%)")
        print(f"Skipped: {skipped}")
        print(f"Failed: {total_samples - total_valid - skipped}")
        
        # Save reports if requested
        if save_csv:
            # Generate summary report CSV
            summary_path = f"{output_dir}{self.dataset_name}_{self.model_name}_validation_summary.csv"
            summary_report = self.generate_summary_report_csv(validation_results, summary_path)
            
            # Generate feature frequency report
            feature_path = f"{output_dir}{self.dataset_name}_{self.model_name}_feature_frequency.csv"
            feature_report = self.generate_feature_frequency_report(feature_path)
            
            # Generate detailed metrics report
            metrics_data = {
                'dataset': self.dataset_name,
                'model': self.model_name,
                'total_samples': total_samples,
                'successful_counterfactuals': successes,
                'total_valid_samples': total_valid,
                'skipped_samples': skipped,
                'success_rate': success_rate
            }
            
            df_metrics = pd.DataFrame([metrics_data])
            metrics_path = f"{output_dir}{self.dataset_name}_{self.model_name}_metrics.csv"
            df_metrics.to_csv(metrics_path, index=False)
            print(f"\nMetrics report saved to: {metrics_path}")
        
        return {
            'validation_results': validation_results,
            'success_rate': success_rate,
            'total_samples': total_samples,
            'successful': successes,
            'total_valid': total_valid,
            'skipped': skipped,
            'feature_frequency': self.feature_frequency,
            'feature_stats': self.feature_stats
        }

    def load_counterfactual_report(self, csv_path):
        """Load counterfactual explanations from CSV file"""
        try:
            cf_report = pd.read_csv(csv_path)
            print(f"Counterfactual report loaded: {cf_report.shape}")
            print(f"Columns: {list(cf_report.columns)}")
            
            # Parse the report into test samples
            test_samples = []
            for _, row in cf_report.iterrows():
                sample = {
                    "instance_id": int(row['Instance_ID']),
                    "actual_class": row['Actual_Class'],
                    "predicted_class": row['Predicted_Class'],
                    "contrast_class": row['Contrast_Class'],
                    "counterfactual_rules": row['Counterfactual_Rules'],
                    "prediction_confidence": float(row['Prediction_Confidence']),
                    "confidence": row['Confidence'],
                    "fidelity": row['Fidelity'],
                    "time_taken": float(row['Time_Taken(s)'])
                }
                test_samples.append(sample)
            
            return test_samples
            
        except Exception as e:
            print(f"Error loading counterfactual report: {e}")
            return None

def validate_single_report(counterfactual_csv_path, output_dir="./validation_results/"):
    """Validate a single counterfactual report"""
    
    print(f"Processing: {counterfactual_csv_path}")
    
    # Initialize validator
    validator = CounterfactualValidator()
    
    # Setup from filename
    if not validator.setup_from_filename(counterfactual_csv_path):
        print(f"Failed to setup validator for {counterfactual_csv_path}")
        return None
    
    # Load data and model
    if not validator.load_data():
        print("Failed to load data")
        return None
    
    if not validator.load_model():
        print("Failed to load model")
        return None
    
    # Load counterfactual report
    test_samples = validator.load_counterfactual_report(counterfactual_csv_path)
    
    if test_samples is None:
        print("Failed to load counterfactual report")
        return None
    
    print(f"Loaded {len(test_samples)} samples from counterfactual report")
    
    # Validate each sample
    validation_results = []
    for sample in test_samples:
        result = validator.validate_counterfactual_sample(
            sample["instance_id"],
            sample["actual_class"],
            sample["predicted_class"],
            sample["contrast_class"],
            sample["counterfactual_rules"],
            sample["prediction_confidence"]
        )
        validation_results.append(result)
    
    # Generate comprehensive report
    report = validator.generate_comprehensive_report(
        validation_results, 
        save_csv=True, 
        output_dir=output_dir
    )
    
    return report

def validate_multiple_reports(report_directory, output_dir="./validation_results/"):
    """Validate multiple counterfactual reports in a directory"""
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all counterfactual report files
    report_files = []
    for file in os.listdir(report_directory):
        if file.endswith('_counterfactual_report.csv'):
            report_files.append(os.path.join(report_directory, file))
    
    print(f"Found {len(report_files)} counterfactual reports to validate")
    
    # Process each report
    all_results = []
    for report_file in report_files:
        print(f"\n{'='*60}")
        result = validate_single_report(report_file, output_dir)
        if result:
            all_results.append({
                'file': os.path.basename(report_file),
                'dataset': result.get('dataset'),
                'model': result.get('model'),
                'results': result
            })
    
    # Generate combined summary
    print(f"\n{'='*60}")
    print("COMBINED VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    combined_data = []
    for result in all_results:
        if result['results']:
            combined_data.append({
                'file': result['file'],
                'dataset': result['results'].get('dataset', 'Unknown'),
                'model': result['results'].get('model', 'Unknown'),
                'total_samples': result['results'].get('total_samples', 0),
                'success_rate': result['results'].get('success_rate', 0),
                'successful': result['results'].get('successful', 0),
                'total_valid': result['results'].get('total_valid', 0),
                'skipped': result['results'].get('skipped', 0)
            })
    
    if combined_data:
        df_combined = pd.DataFrame(combined_data)
        combined_path = f"{output_dir}combined_validation_summary.csv"
        df_combined.to_csv(combined_path, index=False)
        print(f"\nCombined summary saved to: {combined_path}")
        print("\nOverall Results:")
        print(df_combined.to_string(index=False))
    
    return all_results

# Example usage
def main():
    # Example 1: Validate a single report
    # single_report_path = "Evaluation/heart_mlp_counterfactual_report.csv"
    # validate_single_report(single_report_path)
    
    # Example 2: Validate all reports in a directory
    validate_multiple_reports("Evaluation/results/", "./validation_results/")

if __name__ == "__main__":
    main()