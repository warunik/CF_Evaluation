"""
Enhanced FOIL Trees Counterfactual Explanation Generator - MODIFIED VERSION
Uses your specific datasets: Adult, German Credit, Heart Disease, Diabetes, Iris, Breast Cancer, Wine
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from Foil_Trees import domain_mappers, contrastive_explanation
import warnings
warnings.filterwarnings('ignore')

# Define your datasets configuration
ONLINE_DATASETS = {
    "Adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "columns": ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],
        "target_col": 'income'
    },
    "German Credit": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "columns": ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
                   'savings_status', 'employment', 'installment_commitment', 'personal_status',
                   'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans',
                   'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class'],
        "target_col": 'target'
    },
    "Heart Disease": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "columns": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                   'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
        "target_col": 'target'
    },
    "Diabetes": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "columns": ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
                   'bmi', 'diabetes_pedigree', 'age', 'outcome'],
        "target_col": 'outcome'
    }
}

# Define sklearn built-in datasets
SKLEARN_DATASETS = {
    "Iris": {
        "loader": load_iris,
        "target_col": 'target'
    },
    "Breast Cancer": {
        "loader": load_breast_cancer,
        "target_col": 'target'
    },
    "Wine": {
        "loader": load_wine,
        "target_col": 'target'
    }
}

# All available datasets
ALL_DATASETS = {**ONLINE_DATASETS, **SKLEARN_DATASETS}

# Model names mapping
ML_MODELS = {
    'random_forest': 'Random Forest',
    'decision_tree': 'Decision Tree', 
    'logistic_regression': 'Logistic Regression',
    'mlp': 'Multi-Layer Perceptron',
    'xgboost': 'XGBoost'
}

class CounterfactualExplanationGenerator:
    """Enhanced counterfactual explanation generator using pretrained models and preprocessed data"""
    
    def __init__(self, dataset_name, model_name='random_forest', verbose=True):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.label_encoders = {}
        self.explainer = None
        self.metadata = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_features = []
        self.numerical_features = []
        self.class_labels = {}
        
    def load_csv_data(self):
        """Load train/test data from CSV files"""
        try:
            # Construct file paths - adjust these paths according to your directory structure
            train_path = f"data/{self.dataset_name}_train.csv"
            test_path = f"data/{self.dataset_name}_test.csv"
            
            # Alternative path structure if above doesn't work
            if not os.path.exists(train_path):
                train_path = f"{self.dataset_name}_train.csv"
                test_path = f"{self.dataset_name}_test.csv"
            
            # Load the CSV files
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Get target column name
            if self.dataset_name in ONLINE_DATASETS:
                target_col = ONLINE_DATASETS[self.dataset_name]['target_col']
            elif self.dataset_name in SKLEARN_DATASETS:
                target_col = SKLEARN_DATASETS[self.dataset_name]['target_col']
            else:
                # Assume last column is target
                target_col = train_data.columns[-1]
            
            # Split features and target
            self.X_train = train_data.drop(columns=[target_col])
            self.y_train = train_data[target_col]
            self.X_test = test_data.drop(columns=[target_col])
            self.y_test = test_data[target_col]
            
            # Determine feature types automatically
            self.categorical_features = []
            self.numerical_features = []
            
            for col in self.X_train.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(self.X_train[col]):
                    # Check if it has few unique values (might be categorical)
                    unique_vals = self.X_train[col].nunique()
                    if unique_vals <= 10 and self.X_train[col].dtype in ['int64', 'int32']:
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)
            
            # Create class labels mapping
            unique_classes = sorted(self.y_train.unique())
            self.class_labels = {i: str(cls) for i, cls in enumerate(unique_classes)}
            
            if self.verbose:
                print(f"Loaded data for {self.dataset_name}")
                print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
                print(f"Features: {list(self.X_train.columns)}")
                print(f"Categorical features: {self.categorical_features}")
                print(f"Numerical features: {self.numerical_features}")
                print(f"Class labels: {self.class_labels}")
                print(f"Target distribution - Train: {self.y_train.value_counts().to_dict()}")
                
        except Exception as e:
            print(f"Error loading CSV data: {str(e)}")
            print("Please ensure your CSV files are in the correct location:")
            print(f"  - {self.dataset_name}_train.csv")
            print(f"  - {self.dataset_name}_test.csv")
            raise
    
    def load_pretrained_model(self):
        """Load pretrained model from pickle file"""
        try:
            # Try multiple possible file naming conventions
            possible_paths = [
                f"models/{self.dataset_name}_{self.model_name}.pkl",
                f"{self.dataset_name}_{self.model_name}.pkl",
                f"models/{self.model_name}_{self.dataset_name}.pkl",
                f"{self.model_name}_{self.dataset_name}.pkl",
                # Handle spaces in dataset names
                f"models/{self.dataset_name.replace(' ', '_')}_{self.model_name}.pkl",
                f"{self.dataset_name.replace(' ', '_')}_{self.model_name}.pkl",
                f"models/{self.model_name}_{self.dataset_name.replace(' ', '_')}.pkl",
                f"{self.model_name}_{self.dataset_name.replace(' ', '_')}.pkl",
                # Lowercase versions
                f"models/{self.dataset_name.lower()}_{self.model_name}.pkl",
                f"{self.dataset_name.lower()}_{self.model_name}.pkl"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths[:5]}...")
            
            # Check file size first
            file_size = os.path.getsize(model_path)
            if file_size < 100:  # Very small file, likely corrupted
                raise ValueError(f"Model file appears to be corrupted (size: {file_size} bytes)")
            
            # Try different loading methods
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except (pickle.UnpicklingError, ValueError, EOFError) as e:
                print(f"Standard pickle loading failed: {str(e)}")
                # Try with different protocols
                try:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f, encoding='latin1')
                    print("Loaded with latin1 encoding")
                except Exception as e2:
                    print(f"Latin1 encoding failed: {str(e2)}")
                    # Try joblib if pickle fails
                    try:
                        import joblib
                        self.model = joblib.load(model_path)
                        print("Loaded with joblib")
                    except Exception as e3:
                        raise ValueError(f"All loading methods failed. Original error: {str(e)}, Joblib error: {str(e3)}")
                
            if self.verbose:
                print(f"Loaded pretrained model: {ML_MODELS.get(self.model_name, self.model_name)}")
                print(f"Model path: {model_path}")
                print(f"Model type: {type(self.model)}")
                
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            print("Please check:")
            print("1. File exists and is not corrupted")
            print("2. File was saved with compatible pickle/joblib version")
            print("3. File naming convention matches one of these patterns:")
            print(f"   - {self.dataset_name}_{self.model_name}.pkl")
            print(f"   - {self.model_name}_{self.dataset_name}.pkl")
            print(f"   - {self.dataset_name.lower()}_{self.model_name}.pkl")
            raise
    
    def setup_explainer(self):
        """Set up FOIL Trees explainer using loaded data"""
        try:
            class_labels = list(self.class_labels.values())
            
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
            
            if self.verbose:
                print(f"Set up FOIL Trees explainer with {len(class_labels)} classes")
            
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
                n_classes = len(self.class_labels)
                return np.ones((n_samples, n_classes)) / n_classes
        
        return model_predict_wrapper
    
    def generate_counterfactual_explanations(self, n_samples=1000, max_instances=None):
        """Generate counterfactual explanations for test instances"""
        results = []
        
        # Create model wrapper
        model_predict_func = self.create_model_wrapper()
        
        print(f"\n{self.dataset_name.upper()} - COUNTERFACTUAL EXPLANATION REPORT")
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
                actual_class_label = self.class_labels.get(actual_class, str(actual_class))
                predicted_class_label = self.class_labels.get(predicted_class, str(predicted_class))
                
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
        """Extract contrast class from explanation"""
        try:
            contrast_class = "No contrast class extracted"
            if "Contrast Class" in explanation:
                lines = explanation.split('\n')
                for line in lines:
                    if "Contrast Class" in line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            contrast_class = parts[-2].strip()
                            break
            return contrast_class[:500]
        except Exception:
            return "Error extracting contrast class"

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
        """Run complete counterfactual analysis using CSV data and pretrained model"""
        try:
            # Load CSV data
            self.load_csv_data()
            
            # Load pretrained model
            self.load_pretrained_model()
            
            # Setup explainer
            self.setup_explainer()
            
            # Generate counterfactual explanations
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
                    max_instances=max_instances
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
    # Your specific datasets
    datasets_to_analyze = ["German Credit"]
    models_to_use = ['logistic_regression']
    
    print(f"Datasets to analyze: {datasets_to_analyze}")
    print(f"Models to use: {models_to_use}")
    print(f"Total reports to generate: {len(datasets_to_analyze) * len(models_to_use)}")
    
    # Run analysis for all combinations
    all_results = run_analysis_for_datasets_and_models(
        datasets_to_analyze,
        models_to_use,
        n_samples=500,  # Keep this reasonable to avoid memory issues
        max_instances=None  # Process ALL instances
    )
    
    print("\n" + "="*70)
    print("COUNTERFACTUAL EXPLANATION GENERATION COMPLETED")
    print(f"Total instances processed across all reports: {len(all_results)}")
    print("="*70)