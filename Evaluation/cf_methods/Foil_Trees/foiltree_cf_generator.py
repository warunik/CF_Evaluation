"""
Enhanced FOIL Trees Counterfactual Explanation Generator - FIXED VERSION
Uses config file for dataset management and pretrained models
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Foil_Trees import domain_mappers, contrastive_explanation
import warnings
warnings.filterwarnings('ignore')

# Import config
from Evaluation.config import DATASETS, ML_MODELS

class CounterfactualExplanationGenerator:
    """Enhanced counterfactual explanation generator using pretrained models and preprocessed data"""
    
    def __init__(self, dataset_name, model_name='random_forest', verbose=True):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.verbose = verbose
        self.config = DATASETS[dataset_name]
        self.model = None
        self.label_encoders = {}
        self.explainer = None
        self.metadata = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_preprocessed_data(self):
        """Load preprocessed data from saved files"""
        try:
            data_dir = f"{self.dataset_name}"
            
            # Check if files exist
            required_files = ['metadata.json', 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
            for file in required_files:
                if not os.path.exists(f"{data_dir}/{file}"):
                    raise FileNotFoundError(f"Required file not found: {data_dir}/{file}")
            
            # Load metadata
            with open(f"{data_dir}/metadata.json", "r") as f:
                self.metadata = json.load(f)
            
            # Load datasets
            self.X_train = pd.read_csv(f"{data_dir}/X_train.csv")
            self.X_test = pd.read_csv(f"{data_dir}/X_test.csv")
            self.y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]  # Get first column as Series
            self.y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]    # Get first column as Series
            
            # Extract feature types from metadata
            self.categorical_features = self.metadata.get("categorical_features", [])
            self.numerical_features = self.metadata.get("numeric_features", [])
            
            # Validate that we have features
            if not self.categorical_features and not self.numerical_features:
                # Fall back to config feature types
                feature_types = self.config.get("feature_types", {})
                self.categorical_features = [k for k, v in feature_types.items() if v == "categorical"]
                self.numerical_features = [k for k, v in feature_types.items() if v == "numeric"]
            
            # Ensure feature lists contain only columns that exist in the data
            all_columns = set(self.X_train.columns)
            self.categorical_features = [f for f in self.categorical_features if f in all_columns]
            self.numerical_features = [f for f in self.numerical_features if f in all_columns]
            
            if self.verbose:
                print(f"Loaded preprocessed data for {self.config['name']}")
                print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
                print(f"Categorical features: {self.categorical_features}")
                print(f"Numerical features: {self.numerical_features}")
                
        except Exception as e:
            print(f"Error loading preprocessed data: {str(e)}")
            raise
    
    def load_pretrained_model(self):
        """Load pretrained model from saved file"""
        try:
            # Get model path from config
            model_key = model_name_mapping(self.model_name)
            model_path = self.config["model_paths"][model_key]
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            if self.verbose:
                print(f"Loaded pretrained model: {ML_MODELS.get(self.model_name, self.model_name)}")
                print(f"Model path: {model_path}")
                
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            raise
    
    def setup_explainer(self):
        """Set up FOIL Trees explainer using loaded data"""
        try:
            class_labels = list(self.config['class_labels'].values())
            
            # Always use DomainMapperPandas for consistency
            domain_mapper = domain_mappers.DomainMapperPandas(
                train_data=self.X_train,
                contrast_names=class_labels,
                seed=42
            )
            
            # Ensure feature names are set correctly
            domain_mapper.feature_names = list(self.X_train.columns)
            
            self.explainer = contrastive_explanation.ContrastiveExplanation(
                domain_mapper=domain_mapper,
                explanator=contrastive_explanation.TreeExplanator(),
                regression=False,
                verbose=False,
                seed=42
            )
            
        except Exception as e:
            print(f"Error setting up explainer: {str(e)}")
            raise
    
    def create_model_wrapper(self):
        """Create model wrapper for categorical encoding"""
        def model_predict_wrapper(X_batch):
            try:
                # Ensure X_batch is a DataFrame
                if not isinstance(X_batch, pd.DataFrame):
                    if isinstance(X_batch, np.ndarray):
                        X_batch = pd.DataFrame(X_batch, columns=self.X_train.columns)
                    else:
                        # Handle single instance
                        X_batch = pd.DataFrame([X_batch], columns=self.X_train.columns)
                
                X_encoded = X_batch.copy()
                
                # Encode categorical features
                for col in self.categorical_features:
                    if col in X_encoded.columns:
                        # Create label encoder if not already created
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                            # Fit on training data
                            train_values = self.X_train[col].astype(str).fillna('missing')
                            self.label_encoders[col].fit(train_values)
                        
                        # Transform values
                        test_values = X_encoded[col].astype(str).fillna('missing')
                        encoded_values = []
                        
                        for val in test_values:
                            if val in self.label_encoders[col].classes_:
                                encoded_values.append(self.label_encoders[col].transform([val])[0])
                            else:
                                # Handle unseen categories with most frequent class
                                encoded_values.append(0)
                        
                        X_encoded[col] = encoded_values
                
                # Ensure all columns are numeric
                for col in X_encoded.columns:
                    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
                
                # Get predictions
                predictions = self.model.predict_proba(X_encoded)
                
                # Ensure predictions have correct shape
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                
                return predictions
                
            except Exception as e:
                print(f"Error in model wrapper: {str(e)}")
                # Return dummy predictions to avoid crashing
                n_samples = len(X_batch) if hasattr(X_batch, '__len__') else 1
                n_classes = len(self.config['class_labels'])
                return np.ones((n_samples, n_classes)) / n_classes
        
        return model_predict_wrapper
    
    def generate_counterfactual_explanations(self, n_samples=1000, max_instances=None):
        """Generate counterfactual explanations for test instances"""
        results = []
        
        # Create model wrapper
        model_predict_func = self.create_model_wrapper()
        
        print(f"\n{self.config['name'].upper()} - COUNTERFACTUAL EXPLANATION REPORT")
        print("="*70)
        
        # Process ALL test instances if max_instances is None
        if max_instances is None:
            test_indices = self.X_test.index
            sample_size = len(self.X_test)
            print(f"Processing ALL {sample_size} test instances")
        else:
            # Select subset of test instances
            sample_size = min(max_instances, len(self.X_test))
            test_indices = np.random.choice(self.X_test.index, size=sample_size, replace=False)
            print(f"Processing {sample_size} out of {len(self.X_test)} test instances")
        
        processed_count = 0
        error_count = 0
        
        for idx, instance_idx in enumerate(test_indices):
            try:
                # Get instance data
                instance = self.X_test.loc[instance_idx]
                actual_class = self.y_test.loc[instance_idx]
                
                # Get predictions using wrapper
                instance_df = pd.DataFrame([instance], columns=self.X_train.columns)
                prediction_proba = model_predict_func(instance_df)[0]
                predicted_class = np.argmax(prediction_proba)
                
                # Generate explanation
                explanation = self.explainer.explain_instance_domain(
                    model_predict=model_predict_func,
                    fact_sample=instance,
                    foil_method='second',
                    generate_data=True,
                    n_samples=min(n_samples, 500),  # Reduce samples to avoid memory issues
                    include_factual=False
                )
                
                # Parse explanation
                if isinstance(explanation, tuple):
                    main_explanation = str(explanation[0])
                    additional_info = str(explanation[1]) if len(explanation) > 1 else ""
                else:
                    main_explanation = str(explanation)
                    additional_info = ""
                
                # Extract counterfactual rules
                cf_rules = self._extract_counterfactual_rules(main_explanation)
                contrast_class = self._extract_contrast_class(main_explanation)
                confidence = self._extract_confidence(main_explanation)
                fidelity = self._extract_fidelity(main_explanation)
                time_taken = self._extract_time_taken(main_explanation)
                
                # Map class labels
                actual_class_label = self.config['class_labels'].get(actual_class, str(actual_class))
                predicted_class_label = self.config['class_labels'].get(predicted_class, str(predicted_class))
                
                result = {
                    'Instance_ID': instance_idx,
                    'Actual_Class': actual_class_label,
                    'Predicted_Class': predicted_class_label,
                    'Contrast_Class': contrast_class,
                    'Counterfactual_Rules': cf_rules,
                    'Prediction_Confidence': float(np.max(prediction_proba)),
                    'Confidence': confidence,
                    'Fidelity': fidelity,
                    'Time_Taken(s)': time_taken
                }
                
                results.append(result)
                processed_count += 1
                
                # Print progress every 100 instances
                if processed_count % 100 == 0:
                    print(f"Progress: {processed_count}/{sample_size} instances processed")
                
                # Print summary for each instance (optional - disable for large datasets)
                if self.verbose and sample_size <= 100:  # Only show details for small datasets
                    print(f"\nInstance {idx + 1} (ID: {instance_idx}):")
                    print(f"  Actual: {actual_class_label}, Predicted: {predicted_class_label}")
                    print(f"  Confidence: {np.max(prediction_proba):.3f}")
                    print(f"  Counterfactual: {cf_rules}")
                
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"Error processing instance {idx} (ID: {instance_idx}): {str(e)}")
                continue
        
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {processed_count}/{sample_size} instances")
        print(f"Errors encountered: {error_count}")
        
        return results
        
    def _extract_counterfactual_rules(self, explanation):
        """Extract counterfactual rules from explanation"""
        try:
            cf_rules = "No rules extracted"
            if "Counterfactuals" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if "Counterfactuals" in line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            cf_rules = parts[-2].strip()
                            break
            elif "IF" in explanation.upper():
                # Try to extract IF-THEN rules
                lines = explanation.split('\n')
                rule_lines = [line.strip() for line in lines if 'IF' in line.upper() or 'THEN' in line.upper()]
                if rule_lines:
                    cf_rules = '; '.join(rule_lines[:3])  # Take first 3 rules
            
            return cf_rules[:500]  # Truncate very long rules
        except Exception:
            return "Error extracting rules"

    def _extract_contrast_class(self, explanation):
        """Extract counterfactual rules from explanation"""
        try:
            contrast_class = "No rules extracted"
            if "Contrast Class" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if "Contrast Class" in line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            contrast_class = parts[-2].strip()
                            break
            elif "IF" in explanation.upper():
                # Try to extract IF-THEN rules
                lines = explanation.split('\n')
                rule_lines = [line.strip() for line in lines if 'IF' in line.upper() or 'THEN' in line.upper()]
                if rule_lines:
                    contrast_class = '; '.join(rule_lines[:3])  # Take first 3 rules
            
            return contrast_class[:500]  # Truncate very long rules
        except Exception:
            return "Error extracting rules"

    def _extract_confidence(self, explanation):
        """Extract confidence from explanation"""
        try:
            confidence = "No confidence extracted"
            if "Confidence" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if "Confidence" in line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            confidence = parts[-2].strip()
                            break
            return confidence[:500]
        except Exception:
            return "Error extracting confidence"

    def _extract_fidelity(self, explanation):
        """Extract fidelity from explanation"""
        try:
            fidelity = "No fidelity extracted"
            if "Fidelity" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if "Fidelity" in line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            fidelity = parts[-2].strip()
                            break
            return fidelity[:500]
        except Exception:
            return "Error extracting fidelity"

    def _extract_time_taken(self, explanation):
        """Extract time taken (s) from explanation"""
        try:
            time_taken = "No time extracted"
            if "Time Taken" in explanation or "Time Taken(s)" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if ("Time Taken" in line or "Time Taken(s)" in line) and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            time_taken = parts[-2].strip()
                            break
            return time_taken[:500]
        except Exception:
            return "Error extracting time"

    
    def run_full_analysis(self, n_samples=1000, max_instances=None):
        """Run complete counterfactual analysis using preprocessed data and pretrained model"""
        try:
            # Load preprocessed data
            self.load_preprocessed_data()
            
            # Load pretrained model
            self.load_pretrained_model()
            
            # Setup explainer
            self.setup_explainer()
            
            # Generate counterfactual explanations for ALL instances
            results = self.generate_counterfactual_explanations(
                n_samples=n_samples,
                max_instances=max_instances
            )
            
            return results
            
        except Exception as e:
            print(f"Error in full analysis: {str(e)}")
            return []
    
    def save_results(self, results, filename=None):
        """Save results to CSV file"""
        if filename is None:
            filename = f"{self.dataset_name}_{self.model_name}_counterfactual_report.csv"
        
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(filename, index=False)
            
            if self.verbose:
                print(f"\nResults saved to {filename}")
                print(f"Total instances processed: {len(results)}")
                avg_confidence = np.mean([r['Prediction_Confidence'] for r in results])
                print(f"Average confidence: {avg_confidence:.3f}")
        else:
            print(f"No results to save for {filename}")

def model_name_mapping(model_name):
    """Map model names to config keys"""
    mapping = {
        'random_forest': 'random_forest',
        'decision_tree': 'decision_tree',
        'logistic_regression': 'logistic_regression',
        'mlp': 'mlp',
        'xgboost': 'xgboost'
    }
    return mapping.get(model_name, model_name)

def run_analysis_for_datasets_and_models(datasets, models, n_samples=1000, max_instances=None):
    """Run counterfactual analysis for multiple datasets and models"""
    all_results = []
    
    for dataset_name in datasets:
        for model_name in models:
            try:
                print(f"\n{'='*70}")
                print(f"ANALYZING {dataset_name.upper()} DATASET WITH {model_name.upper()} MODEL")
                print('='*70)
                
                generator = CounterfactualExplanationGenerator(dataset_name, model_name)
                results = generator.run_full_analysis(
                    n_samples=n_samples,
                    max_instances=max_instances  # Pass None to process all instances
                )
                
                # Save results with model-specific filename
                filename = f"{dataset_name}_{model_name}_counterfactual_report.csv"
                generator.save_results(results, filename)
                
                if results:
                    print(f"\n{dataset_name.upper()} - {model_name.upper()} SUMMARY:")
                    print(f"Total instances processed: {len(results)}")
                    avg_confidence = np.mean([r['Prediction_Confidence'] for r in results])
                    print(f"Average confidence: {avg_confidence:.3f}")
                
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error processing {dataset_name} with {model_name}: {str(e)}")
                continue
    
    return all_results

if __name__ == "__main__":
    # Available datasets and models
    available_datasets = list(DATASETS.keys())
    available_models = list(ML_MODELS.keys())
    
    # Verify we have 5 datasets and 5 models
    datasets_to_analyze = available_datasets[:5]
    models_to_use = available_models[:5]
    
    print(f"Datasets to analyze: {datasets_to_analyze}")
    print(f"Models to use: {models_to_use}")
    print(f"Total reports to generate: {len(datasets_to_analyze) * len(models_to_use)}")
    
    # Run analysis for all combinations - PROCESS ALL INSTANCES
    all_results = run_analysis_for_datasets_and_models(
        datasets_to_analyze,
        models_to_use,
        n_samples=500,  # Keep this reasonable to avoid memory issues
        max_instances=None  # Changed to None to process ALL instances
    )
    
    print("\n" + "="*70)
    print("COUNTERFACTUAL EXPLANATION GENERATION COMPLETED")
    print(f"Total instances processed across all reports: {len(all_results)}")
    print("="*70)