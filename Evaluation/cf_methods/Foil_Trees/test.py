"""
Enhanced FOIL Trees Counterfactual Explanation Generator
Uses config file for dataset management and generates counterfactual explanations
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import config
from Evaluation.config import DATASETS, ML_MODELS

# Your FOIL Trees imports
from Foil_Trees import domain_mappers, contrastive_explanation

class CounterfactualExplanationGenerator:
    """Enhanced counterfactual explanation generator using config file"""
    
    def __init__(self, dataset_name, model_name='random_forest', verbose=True):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.verbose = verbose
        self.config = DATASETS[dataset_name]
        self.model = None
        self.label_encoders = {}
        self.explainer = None
        
    def load_and_prepare_data(self):
        """Load and prepare dataset based on config"""
        # Load dataset
        df = pd.read_csv(self.config['path'])
        
        # Drop columns if specified
        if 'drop_columns' in self.config:
            df = df.drop(columns=self.config['drop_columns'], errors='ignore')
        
        # Prepare features and target
        X = df.drop(self.config['target_column'], axis=1)
        y = df[self.config['target_column']]
        
        # Handle missing values
        self._handle_missing_values(X)
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if col in self.config['feature_types']:
                if self.config['feature_types'][col] == 'categorical':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                # Auto-detect if not in config
                if X[col].dtype == 'object':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        if self.verbose:
            print(f"Dataset: {self.config['name']}")
            print(f"Shape: {X.shape}")
            print(f"Categorical features: {categorical_features}")
            print(f"Numerical features: {numerical_features}")
        
        return X, y
    
    def _handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
    
    def _get_model(self):
        """Get ML model based on config"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'mlp': MLPClassifier(random_state=42, max_iter=1000),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        return models.get(self.model_name, RandomForestClassifier(n_estimators=100, random_state=42))
    
    def prepare_model_data(self, X_train, X_test):
        """Prepare data for model training (encode categorical variables)"""
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        # Encode categorical variables for sklearn model
        for col in self.categorical_features:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        return X_train_encoded, X_test_encoded
    
    def train_model(self, X_train, y_train):
        """Train the ML model"""
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        if self.verbose:
            print(f"Model trained: {ML_MODELS.get(self.model_name, self.model_name)}")
    
    def setup_explainer(self, X_train):
        """Set up FOIL Trees explainer"""
        class_labels = list(self.config['class_labels'].values())
        
        # Choose appropriate domain mapper
        if self.categorical_features:
            domain_mapper = domain_mappers.DomainMapperPandas(
                train_data=X_train,
                contrast_names=class_labels,
                seed=42
            )
        else:
            domain_mapper = domain_mappers.DomainMapperTabular(
                train_data=X_train.values,
                feature_names=list(X_train.columns),
                contrast_names=class_labels,
                categorical_features=None,
                seed=42
            )
        
        domain_mapper.feature_names = list(X_train.columns)
        
        self.explainer = contrastive_explanation.ContrastiveExplanation(
            domain_mapper=domain_mapper,
            explanator=contrastive_explanation.TreeExplanator(),
            regression=False,
            verbose=False,
            seed=42
        )
    
    def create_model_wrapper(self, X_train):
        """Create model wrapper for categorical encoding"""
        def model_predict_wrapper(X_batch):
            if isinstance(X_batch, pd.DataFrame):
                X_encoded = X_batch.copy()
            else:
                X_encoded = pd.DataFrame(X_batch, columns=X_train.columns)
            
            # Encode categorical features
            for col in self.categorical_features:
                if col in X_encoded.columns:
                    try:
                        X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        X_encoded[col] = X_encoded[col].map(
                            lambda x: self.label_encoders[col].transform([str(x)])[0] 
                            if str(x) in self.label_encoders[col].classes_ else 0
                        )
            
            return self.model.predict_proba(X_encoded)
        
        return model_predict_wrapper
    
    def generate_counterfactual_explanations(self, X_test, y_test, X_train, n_samples=1000):
        """Generate counterfactual explanations for all test instances"""
        results = []
        
        # Prepare encoded test data for predictions
        if self.categorical_features:
            X_test_encoded = X_test.copy()
            for col in self.categorical_features:
                X_test_encoded[col] = self.label_encoders[col].transform(X_test_encoded[col].astype(str))
        else:
            X_test_encoded = X_test
        
        # Create model wrapper
        if self.categorical_features:
            model_predict_func = self.create_model_wrapper(X_train)
        else:
            model_predict_func = self.model.predict_proba
        
        print(f"\n{self.config['name'].upper()} - COUNTERFACTUAL EXPLANATION REPORT")
        print("="*70)
        
        for idx, (test_idx, instance) in enumerate(X_test.iterrows()):
            try:
                # Get predictions
                instance_encoded = X_test_encoded.iloc[idx]
                actual_class = y_test.iloc[idx]
                predicted_class = self.model.predict([instance_encoded])[0]
                prediction_proba = self.model.predict_proba([instance_encoded])[0]
                
                # Prepare instance for explanation
                if self.categorical_features:
                    fact_sample = instance
                else:
                    fact_sample = instance.values
                
                # Generate explanation
                explanation = self.explainer.explain_instance_domain(
                    model_predict=model_predict_func,
                    fact_sample=fact_sample,
                    foil_method='second',
                    generate_data=True,
                    n_samples=n_samples,
                    include_factual=True
                )
                
                # Parse explanation
                if isinstance(explanation, tuple):
                    main_explanation = explanation[0]
                    additional_info = explanation[1] if len(explanation) > 1 else ""
                else:
                    main_explanation = str(explanation)
                    additional_info = ""
                
                # Extract counterfactual rules
                cf_rules = self._extract_counterfactual_rules(main_explanation)
                
                # Map class labels
                actual_class_label = self.config['class_labels'].get(actual_class, actual_class)
                predicted_class_label = self.config['class_labels'].get(predicted_class, predicted_class)
                
                result = {
                    'Instance_ID': test_idx,
                    'Actual_Class': actual_class_label,
                    'Predicted_Class': predicted_class_label,
                    'Prediction_Confidence': max(prediction_proba),
                    'Counterfactual_Rules': cf_rules,
                    'Full_Explanation': main_explanation,
                    'Additional_Info': additional_info
                }
                
                results.append(result)
                
                # Print summary for each instance
                if self.verbose:
                    print(f"\nInstance {idx + 1} (ID: {test_idx}):")
                    print(f"  Actual: {actual_class_label}, Predicted: {predicted_class_label}")
                    print(f"  Confidence: {max(prediction_proba):.3f}")
                    print(f"  Counterfactual: {cf_rules}")
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing instance {idx}: {str(e)}")
                continue
        
        return results
    
    def _extract_counterfactual_rules(self, explanation):
        """Extract counterfactual rules from explanation"""
        cf_rules = "No rules extracted"
        if "Counterfactuals" in explanation:
            lines = explanation.split('\n')
            for line in lines:
                if "Counterfactuals" in line:
                    cf_rules = line.split('|')[-2].strip()
                    break
        return cf_rules
    
    def run_full_analysis(self, test_size=0.3, n_samples=1000):
        """Run complete counterfactual analysis"""
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Prepare model data (encode categorical variables)
        X_train_encoded, X_test_encoded = self.prepare_model_data(X_train, X_test)
        
        # Train model
        self.train_model(X_train_encoded, y_train)
        
        # Setup explainer
        self.setup_explainer(X_train)
        
        # Generate counterfactual explanations
        results = self.generate_counterfactual_explanations(
            X_test, y_test, X_train, n_samples=n_samples
        )
        
        return results
    
    def save_results(self, results, filename=None):
        """Save results to CSV file"""
        if filename is None:
            filename = f"{self.dataset_name}_{self.model_name}_counterfactual_report.csv"
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        
        if self.verbose:
            print(f"\nResults saved to {filename}")
            print(f"Total instances processed: {len(results)}")
            if results:
                print(f"Average confidence: {np.mean([r['Prediction_Confidence'] for r in results]):.3f}")

def run_analysis_for_dataset(dataset_name, model_name='random_forest', n_samples=1000):
    """Run counterfactual analysis for a specific dataset"""
    try:
        generator = CounterfactualExplanationGenerator(dataset_name, model_name)
        results = generator.run_full_analysis(n_samples=n_samples)
        generator.save_results(results)
        return results
    except Exception as e:
        print(f"Error processing {dataset_name} dataset: {str(e)}")
        return []

if __name__ == "__main__":
    # Available datasets from config
    available_datasets = list(DATASETS.keys())
    print("Available datasets:", available_datasets)
    
    # Run analysis for specific datasets
    datasets_to_analyze = ['diabetes', 'adult', 'heart', 'bank', 'german']  # Add more as needed
    
    for dataset_name in datasets_to_analyze:
        if dataset_name in available_datasets:
            print(f"\n{'='*70}")
            print(f"ANALYZING {dataset_name.upper()} DATASET")
            print('='*70)
            
            results = run_analysis_for_dataset(dataset_name, model_name='random_forest')
            
            if results:
                print(f"\n{dataset_name.upper()} DATASET SUMMARY:")
                print(f"Total instances processed: {len(results)}")
                print(f"Average confidence: {np.mean([r['Prediction_Confidence'] for r in results]):.3f}")
        else:
            print(f"Dataset {dataset_name} not found in config")
    
    print("\n" + "="*70)
    print("COUNTERFACTUAL EXPLANATION GENERATION COMPLETED")
    print("="*70)