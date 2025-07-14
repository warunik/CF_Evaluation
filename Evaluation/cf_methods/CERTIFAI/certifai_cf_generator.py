#!/usr/bin/env python3
"""
CERTIFAI Batch Counterfactual Generator
Runs counterfactual generation for all datasets and models from config
Enhanced version with detailed CSV reports
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from Evaluation.cf_methods.CERTIFAI.certifai import CERTIFAI
    from Evaluation.config import DATASETS, ML_MODELS
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure CERTIFAI is installed and config.py is in the correct location.")
    sys.exit(1)


class CERTIFAIBatchGenerator:
    """
    Enhanced Batch CERTIFAI Counterfactual Generator with detailed reporting
    """
    
    def __init__(self, generations=5, final_k=3, num_samples=10):
        """
        Initialize the batch generator
        
        Args:
            generations (int): Number of generations for genetic algorithm
            final_k (int): Number of counterfactuals to keep per sample
            num_samples (int): Number of samples to process per dataset
        """
        self.generations = generations
        self.final_k = final_k
        self.num_samples = num_samples
        
        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Track overall progress
        self.total_combinations = len(DATASETS) * len(ML_MODELS)
        self.completed = 0
        self.failed = 0
        self.summary_data = []
    
    def load_preprocessed_data(self, dataset_name):
        """Load preprocessed data if available, otherwise preprocess raw data"""
        dataset_config = DATASETS[dataset_name]
        preprocessed_dir = dataset_config['preprocessed_dir']
        
        # Check if all required files exist
        required_files = [
            f"{preprocessed_dir}/X_train.csv",
            f"{preprocessed_dir}/X_test.csv", 
            f"{preprocessed_dir}/y_train.csv",
            f"{preprocessed_dir}/y_test.csv"
        ]
        
        if all(os.path.exists(file) for file in required_files):
            print(f"Loading preprocessed data for {dataset_name}...")
            
            # Load feature data
            X_train = pd.read_csv(f"{preprocessed_dir}/X_train.csv")
            X_test = pd.read_csv(f"{preprocessed_dir}/X_test.csv")
            
            # Load target data - handle both single column and multiple column cases
            y_train_df = pd.read_csv(f"{preprocessed_dir}/y_train.csv")
            y_test_df = pd.read_csv(f"{preprocessed_dir}/y_test.csv")
            
            # Extract target values - handle different CSV formats
            if len(y_train_df.columns) == 1:
                # Single column case
                y_train = y_train_df.iloc[:, 0].values
                y_test = y_test_df.iloc[:, 0].values
            else:
                # Multiple columns case - take the target column or last column
                target_column = dataset_config.get('target_column', y_train_df.columns[-1])
                if target_column in y_train_df.columns:
                    y_train = y_train_df[target_column].values
                    y_test = y_test_df[target_column].values
                else:
                    y_train = y_train_df.iloc[:, -1].values
                    y_test = y_test_df.iloc[:, -1].values
            
            # Load scaler if available
            scaler_path = f"{preprocessed_dir}/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = None
            
            # Load metadata if available
            metadata_path = f"{preprocessed_dir}/metadata.json"
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded metadata: {metadata}")
            
            return X_train, X_test, y_train, y_test, scaler
        
        else:
            print(f"Preprocessed data not found. Processing raw data for {dataset_name}...")
            print(f"Missing files: {[f for f in required_files if not os.path.exists(f)]}")
            return self.preprocess_raw_data(dataset_name)
    
    def preprocess_raw_data(self, dataset_name):
        """
        Preprocess raw data for a dataset
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        dataset_config = DATASETS[dataset_name]
        
        # Load raw data
        df = pd.read_csv(dataset_config['path'])
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        target_col = dataset_config['target_column']
        if 'drop_columns' in dataset_config:
            df = df.drop(columns=dataset_config['drop_columns'])
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables
        if 'feature_types' in dataset_config:
            categorical_cols = [col for col, dtype in dataset_config['feature_types'].items() 
                              if dtype == 'categorical' and col in X.columns]
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable if needed
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns
        )
        
        # Save preprocessed data
        os.makedirs(dataset_config['preprocessed_dir'], exist_ok=True)
        X_train_scaled.to_csv(f"{dataset_config['preprocessed_dir']}/X_train.csv", index=False)
        X_test_scaled.to_csv(f"{dataset_config['preprocessed_dir']}/X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{dataset_config['preprocessed_dir']}/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{dataset_config['preprocessed_dir']}/y_test.csv", index=False)
        
        # Save scaler
        with open(f"{dataset_config['preprocessed_dir']}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def load_model(self, dataset_name, model_name):
        """
        Load a trained model
        
        Args:
            dataset_name (str): Name of the dataset
            model_name (str): Name of the model
            
        Returns:
            object: Trained model
        """
        dataset_config = DATASETS[dataset_name]
        model_path = dataset_config['model_paths'][model_name]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def calculate_fidelity(self, model, X_sample, counterfactual, prediction_changed):
        """
        Calculate fidelity score for a counterfactual
        
        Args:
            model: Trained model
            X_sample: Original sample
            counterfactual: Counterfactual sample
            prediction_changed: Whether prediction changed
            
        Returns:
            float: Fidelity score
        """
        # Simple fidelity calculation - can be enhanced based on requirements
        if prediction_changed:
            return 1.0  # High fidelity if prediction changed
        else:
            return 0.0  # Low fidelity if prediction didn't change
    
    def generate_counterfactual_rules(self, original_sample, counterfactual, feature_names, threshold=1e-6):
        """
        Generate human-readable rules for counterfactual changes
        
        Args:
            original_sample: Original sample values
            counterfactual: Counterfactual sample values
            feature_names: Names of features
            threshold: Minimum change threshold
            
        Returns:
            str: Human-readable rules
        """
        rules = []
        
        for i, feature_name in enumerate(feature_names):
            original_val = original_sample[i]
            cf_val = counterfactual[i]
            
            if abs(original_val - cf_val) > threshold:
                if cf_val > original_val:
                    direction = "increase"
                    change = cf_val - original_val
                else:
                    direction = "decrease"
                    change = original_val - cf_val
                
                rules.append(f"{feature_name} {direction} by {change:.4f}")
        
        return "; ".join(rules) if rules else "No significant changes"
    
    def generate_counterfactuals_for_combination(self, dataset_name, model_name):
        """
        Generate counterfactuals for a specific dataset-model combination
        
        Args:
            dataset_name (str): Name of the dataset
            model_name (str): Name of the model
            
        Returns:
            dict: Results or None if failed
        """
        print(f"\nProcessing: {dataset_name} + {model_name}")
        print("-" * 50)
        
        try:
            # Load data and model
            X_train, X_test, y_train, y_test, scaler = self.load_preprocessed_data(dataset_name)
            model = self.load_model(dataset_name, model_name)
            
            # Select samples for counterfactual generation
            num_samples = min(self.num_samples, len(X_test))
            sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
            X_samples = X_test.iloc[sample_indices]
            
            print(f"Generating counterfactuals for {num_samples} samples...")
            
            # Initialize CERTIFAI
            certifai = CERTIFAI(Pm=0.2, Pc=0.5, numpy_dataset=X_train.values)
            
            # Custom model wrapper for sklearn models
            class SklearnModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict(self, X):
                    if hasattr(X, 'values'):
                        return self.model.predict(X.values)
                    return self.model.predict(X)
                
                def predict_proba(self, X):
                    if hasattr(X, 'values'):
                        return self.model.predict_proba(X.values)
                    return self.model.predict_proba(X)
            
            # Wrap the model
            model_wrapper = SklearnModelWrapper(model)
            
            # Configure CERTIFAI
            certifai.set_distance(kind='automatic', x=X_samples)
            certifai.set_population(X_samples)
            certifai.set_constraints(X_samples)
            
            # Record timing
            start_time = time.time()
            
            # Generate counterfactuals
            certifai.fit(
                model=model_wrapper,
                x=X_samples,
                pytorch=False,
                classification=True,
                generations=self.generations,
                final_k=self.final_k,
                normalisation='standard' if scaler else None,
                verbose=False
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Process results
            results = self.process_results_for_report(
                certifai, sample_indices, X_test, y_test, model, 
                dataset_name, model_name, total_time
            )
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def process_results_for_report(self, certifai, sample_indices, X_test, y_test, model, 
                                  dataset_name, model_name, total_time):
        """
        Process and format the counterfactual results for detailed reporting
        
        Args:
            certifai: CERTIFAI instance
            sample_indices: Indices of samples processed
            X_test: Test data
            y_test: Test labels
            model: Trained model
            dataset_name: Name of dataset
            model_name: Name of model
            total_time: Total time taken for generation
            
        Returns:
            dict: Processed results
        """
        if not hasattr(certifai, 'results') or not certifai.results:
            return None
        
        report_data = []
        dataset_config = DATASETS[dataset_name]
        class_labels = dataset_config.get('class_labels', {})
        
        for i, (original, counterfactuals, distances) in enumerate(certifai.results):
            sample_idx = sample_indices[i]
            
            # Original sample info
            original_sample = X_test.iloc[sample_idx:sample_idx+1]
            original_values = original_sample.values[0]
            actual_class = y_test[sample_idx]
            original_pred = model.predict(original_sample)[0]
            
            try:
                original_prob = model.predict_proba(original_sample)[0]
                original_confidence = np.max(original_prob)
            except:
                original_prob = None
                original_confidence = None
            
            # Process each counterfactual
            for j, (cf, distance) in enumerate(zip(counterfactuals, distances)):
                cf_df = pd.DataFrame([cf], columns=original_sample.columns)
                cf_pred = model.predict(cf_df)[0]
                
                try:
                    cf_prob = model.predict_proba(cf_df)[0]
                    cf_confidence = np.max(cf_prob)
                except:
                    cf_prob = None
                    cf_confidence = None
                
                # Calculate prediction change
                prediction_changed = cf_pred != original_pred
                
                # Calculate fidelity
                fidelity = self.calculate_fidelity(model, original_values, cf, prediction_changed)
                
                # Generate counterfactual rules
                cf_rules = self.generate_counterfactual_rules(
                    original_values, cf, original_sample.columns
                )
                
                # Create report row
                report_row = {
                    'Instance_ID': f"{dataset_name}_{sample_idx}_{j}",
                    'Actual_Class': class_labels.get(actual_class, f'Class_{actual_class}'),
                    'Predicted_Class': class_labels.get(original_pred, f'Class_{original_pred}'),
                    'Contrast_Class': class_labels.get(cf_pred, f'Class_{cf_pred}'),
                    'Counterfactual_Rules': cf_rules,
                    'Prediction_Confidence': original_confidence,
                    'Confidence': cf_confidence,
                    'Fidelity': fidelity,
                    'Time_Taken(s)': total_time / len(certifai.results),  # Average time per sample
                    'Distance': distance,
                    'Prediction_Changed': prediction_changed,
                    'Sample_Index': sample_idx,
                    'CF_Index': j
                }
                
                # Add individual feature values if needed
                for k, feature_name in enumerate(original_sample.columns):
                    report_row[f'Original_{feature_name}'] = original_values[k]
                    report_row[f'CF_{feature_name}'] = cf[k]
                
                report_data.append(report_row)
        
        return {
            'report_data': report_data,
            'dataset': dataset_name,
            'model': model_name,
            'total_samples': len(sample_indices),
            'total_counterfactuals': len(report_data),
            'total_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_detailed_report(self, results, dataset_name, model_name):
        """
        Save detailed CSV report in the requested format
        
        Args:
            results (dict): Results to save
            dataset_name (str): Name of dataset
            model_name (str): Name of model
            
        Returns:
            str: Path to saved report
        """
        if not results or not results['report_data']:
            return None
        
        # Create DataFrame from report data
        df = pd.DataFrame(results['report_data'])
        
        # Reorder columns to match requested format
        base_columns = [
            'Instance_ID', 'Actual_Class', 'Predicted_Class', 'Contrast_Class',
            'Counterfactual_Rules', 'Prediction_Confidence', 'Confidence',
            'Fidelity', 'Time_Taken(s)', 'Distance', 'Prediction_Changed'
        ]
        
        # Add any additional columns that exist
        additional_columns = [col for col in df.columns if col not in base_columns]
        final_columns = base_columns + additional_columns
        
        # Select only existing columns
        available_columns = [col for col in final_columns if col in df.columns]
        df_final = df[available_columns]
        
        # Save report
        report_filename = f"{dataset_name}_{model_name}_counterfactual_report.csv"
        report_path = os.path.join(self.results_dir, report_filename)
        df_final.to_csv(report_path, index=False)
        
        print(f"Detailed report saved: {report_path}")
        print(f"Report contains {len(df_final)} counterfactual instances")
        
        return report_path
    
    def display_summary(self, results, dataset_name, model_name):
        """
        Display summary of results
        
        Args:
            results (dict): Results to display
            dataset_name (str): Name of dataset
            model_name (str): Name of model
        """
        if not results or not results['report_data']:
            print("No results to display")
            return
        
        print(f"\nSUMMARY - {dataset_name} + {model_name}")
        print("-" * 40)
        
        df = pd.DataFrame(results['report_data'])
        
        total_samples = results['total_samples']
        total_counterfactuals = results['total_counterfactuals']
        
        print(f"Samples processed: {total_samples}")
        print(f"Counterfactuals generated: {total_counterfactuals}")
        print(f"Total time: {results['total_time']:.2f}s")
        
        if total_counterfactuals > 0:
            print(f"Avg counterfactuals per sample: {total_counterfactuals/total_samples:.2f}")
            print(f"Avg time per sample: {results['total_time']/total_samples:.2f}s")
            
            # Calculate statistics
            prediction_changes = df['Prediction_Changed'].sum()
            avg_distance = df['Distance'].mean()
            avg_fidelity = df['Fidelity'].mean()
            
            print(f"Prediction flip rate: {prediction_changes/total_counterfactuals:.2%}")
            print(f"Average distance: {avg_distance:.4f}")
            print(f"Average fidelity: {avg_fidelity:.4f}")
            
            if 'Confidence' in df.columns:
                avg_confidence = df['Confidence'].mean()
                print(f"Average CF confidence: {avg_confidence:.4f}")
    
    def run_all_combinations(self):
        """
        Run counterfactual generation for all dataset-model combinations
        """
        print("Starting CERTIFAI Batch Counterfactual Generation")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Generations: {self.generations}")
        print(f"  Final K: {self.final_k}")
        print(f"  Samples per dataset: {self.num_samples}")
        print(f"  Total combinations: {self.total_combinations}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Process all combinations
        for dataset_name in DATASETS.keys():
            for model_name in ML_MODELS.keys():
                try:
                    results = self.generate_counterfactuals_for_combination(
                        dataset_name, model_name
                    )
                    
                    if results:
                        # Save detailed report
                        report_path = self.save_detailed_report(results, dataset_name, model_name)
                        
                        # Display summary
                        self.display_summary(results, dataset_name, model_name)
                        
                        # Add to overall summary
                        self.summary_data.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'status': 'SUCCESS',
                            'samples_processed': results['total_samples'],
                            'total_counterfactuals': results['total_counterfactuals'],
                            'total_time': results['total_time'],
                            'avg_time_per_sample': results['total_time'] / results['total_samples'],
                            'report_path': report_path,
                            'timestamp': results['timestamp']
                        })
                        
                        self.completed += 1
                        
                    else:
                        print("Failed to generate counterfactuals")
                        self.summary_data.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'status': 'FAILED',
                            'error': 'No results generated',
                            'timestamp': datetime.now().isoformat()
                        })
                        self.failed += 1
                        
                except Exception as e:
                    print(f"Error processing {dataset_name} + {model_name}: {e}")
                    self.summary_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'status': 'ERROR',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.failed += 1
                
                # Progress update
                total_processed = self.completed + self.failed
                print(f"Progress: {total_processed}/{self.total_combinations} combinations completed")
        
        # Save overall summary
        self.save_final_summary(start_time)
    
    def save_final_summary(self, start_time):
        """
        Save final summary of all results
        
        Args:
            start_time: Start time of the batch process
        """
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.summary_data)
        summary_path = os.path.join(self.results_dir, "batch_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL BATCH SUMMARY")
        print("=" * 60)
        print(f"Total combinations: {self.total_combinations}")
        print(f"Successful: {self.completed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {(self.completed/self.total_combinations)*100:.1f}%")
        print(f"Total duration: {duration}")
        print(f"Summary saved: {summary_path}")
        print(f"Individual reports saved in: {self.results_dir}/")
        
        # Success by dataset/model
        if self.completed > 0:
            success_df = summary_df[summary_df['status'] == 'SUCCESS']
            if not success_df.empty:
                print(f"\nGenerated reports:")
                for _, row in success_df.iterrows():
                    print(f"  {row['dataset']}_{row['model']}_counterfactual_report.csv")
                    if 'total_time' in row:
                        print(f"    - {row['samples_processed']} samples, {row['total_counterfactuals']} counterfactuals")
                        print(f"    - Time: {row['total_time']:.2f}s (avg: {row['avg_time_per_sample']:.2f}s per sample)")
        
        print("=" * 60)


def main():
    """
    Main function to run batch counterfactual generation
    """
    # Configuration - adjust these parameters as needed
    GENERATIONS = 5      # Number of generations for genetic algorithm
    FINAL_K = 3         # Number of counterfactuals to keep per sample
    NUM_SAMPLES = 10    # Number of samples to process per dataset
    
    # Initialize and run batch generator
    generator = CERTIFAIBatchGenerator(
        generations=GENERATIONS,
        final_k=FINAL_K,
        num_samples=NUM_SAMPLES
    )
    
    generator.run_all_combinations()


if __name__ == "__main__":
    main()