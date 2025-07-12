import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Enhanced dataset configurations
DATASETS = {
    "heart": {
        "name": "Heart Disease",
        "path": "Evaluation/data/heart.csv",
        "target_column": "target",
        "class_labels": {0: "No Heart Disease", 1: "Heart Disease"},
        "feature_types": {
            "age": "numeric",
            "sex": "numeric",
            "cp": "numeric",
            "trestbps": "numeric",
            "chol": "numeric",
            "fbs": "numeric",
            "restecg": "numeric",
            "thalach": "numeric",
            "exang": "numeric",
            "oldpeak": "numeric",
            "slope": "numeric",
            "ca": "numeric",
            "thal": "numeric"
        }
    },
    "diabetes": {
        "name": "Diabetes Prediction",
        "path": "Evaluation/data/diabetes.csv",
        "target_column": "Outcome",
        "class_labels": {0: "No Diabetes", 1: "Diabetes"},
        "feature_types": {
            "Pregnancies": "numeric",
            "Glucose": "numeric",
            "BloodPressure": "numeric",
            "SkinThickness": "numeric",
            "Insulin": "numeric",
            "BMI": "numeric",
            "DiabetesPedigreeFunction": "numeric",
            "Age": "numeric"
        }
    },
    "adult": {
        "name": "Income Prediction",
        "path": "Evaluation/data/adult.csv",
        "target_column": "class",
        "drop_columns": [" fnlwgt", " education-num", " native-country"],
        "class_labels": {0: "<=50K", 1: ">50K"},
        "feature_types": {
            "age": "numeric",
            "workclass": "categorical",
            "fnlwgt": "numeric",
            "education": "categorical",
            "education-num": "numeric",
            "marital-status": "categorical",
            "occupation": "categorical",
            "relationship": "categorical",
            "race": "categorical",
            "sex": "categorical",
            "capital-gain": "numeric",
            "capital-loss": "numeric",
            "hours-per-week": "numeric",
            "native-country": "categorical"
        }
    },
    "bank": {
        "name": "Credit Approval",
        "path": "Evaluation/data/bank.csv",
        "target_column": "give_credit",
        "class_labels": {0: "Deny Credit", 1: "Approve Credit"},
        "feature_types": {
            "revolving": "numeric",
            "age": "numeric",
            "nbr_30_59_days_past_due_not_worse": "numeric",
            "debt_ratio": "numeric",
            "monthly_income": "numeric",
            "nbr_open_credits_and_loans": "numeric",
            "nbr_90_days_late": "numeric",
            "nbr_real_estate_loans_or_lines": "numeric",
            "nbr_60_89_days_past_due_not_worse": "numeric",
            "dependents": "numeric"
        }
    },
    "german": {
        "name": "German Credit Risk",
        "path": "Evaluation/data/german_credit.csv",
        "target_column": "default",
        "class_labels": {0: "Good Credit", 1: "Bad Credit"},
        "feature_types": {
            "account_check_status": "categorical",
            "duration_in_month": "numeric",
            "credit_history": "categorical",
            "purpose": "categorical",
            "credit_amount": "numeric",
            "savings": "categorical",
            "present_emp_since": "categorical",
            "installment_as_income_perc": "numeric",
            "personal_status_sex": "categorical",
            "other_debtors": "categorical",
            "present_res_since": "numeric",
            "property": "categorical",
            "age": "numeric",
            "other_installment_plans": "categorical",
            "housing": "categorical",
            "credits_this_bank": "numeric",
            "job": "categorical",
            "people_under_maintenance": "numeric",
            "telephone": "categorical",
            "foreign_worker": "categorical"
        }
    }
}

# Create comprehensive output directory structure
def create_output_directories():
    """Create all necessary output directories"""
    directories = [
        "saved_models",
        "reports", 
        "split_datasets",
        "split_datasets/csv",
        "split_datasets/npy",
        "split_datasets/pickle"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created output directories")

def save_split_datasets(dataset_name, X_train, X_test, y_train, y_test, config):
    """Save split datasets in multiple formats with comprehensive metadata"""
    
    # Create dataset-specific directories
    for format_type in ['csv', 'npy', 'pickle']:
        os.makedirs(f"split_datasets/{format_type}/{dataset_name}", exist_ok=True)
    
    # Convert to consistent format for saving
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    y_train_series = pd.Series(y_train, name='target') if not isinstance(y_train, pd.Series) else y_train
    y_test_series = pd.Series(y_test, name='target') if not isinstance(y_test, pd.Series) else y_test
    
    # 1. Save as CSV (human-readable, widely compatible)
    csv_dir = f"split_datasets/csv/{dataset_name}"
    X_train_df.to_csv(f"{csv_dir}/X_train.csv", index=False)
    X_test_df.to_csv(f"{csv_dir}/X_test.csv", index=False)
    y_train_series.to_csv(f"{csv_dir}/y_train.csv", index=False)
    y_test_series.to_csv(f"{csv_dir}/y_test.csv", index=False)
    
    # 2. Save as NumPy arrays (efficient storage, fast loading)
    npy_dir = f"split_datasets/npy/{dataset_name}"
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values
    y_train_np = y_train_series.values
    y_test_np = y_test_series.values
    
    np.save(f"{npy_dir}/X_train.npy", X_train_np)
    np.save(f"{npy_dir}/X_test.npy", X_test_np)
    np.save(f"{npy_dir}/y_train.npy", y_train_np)
    np.save(f"{npy_dir}/y_test.npy", y_test_np)
    
    # 3. Save as pickle (preserves exact Python objects)
    pickle_dir = f"split_datasets/pickle/{dataset_name}"
    with open(f"{pickle_dir}/X_train.pkl", 'wb') as f:
        pickle.dump(X_train_df, f)
    with open(f"{pickle_dir}/X_test.pkl", 'wb') as f:
        pickle.dump(X_test_df, f)
    with open(f"{pickle_dir}/y_train.pkl", 'wb') as f:
        pickle.dump(y_train_series, f)
    with open(f"{pickle_dir}/y_test.pkl", 'wb') as f:
        pickle.dump(y_test_series, f)
    
    # Create comprehensive metadata
    feature_names = list(X_train_df.columns)
    train_class_dist = dict(y_train_series.value_counts().sort_index())
    test_class_dist = dict(y_test_series.value_counts().sort_index())
    
    metadata = {
        'dataset_info': {
            'name': config['name'],
            'dataset_key': dataset_name,
            'target_column': config['target_column'],
            'class_labels': config['class_labels']
        },
        'split_info': {
            'random_state': SEED,
            'test_size': 0.3,
            'stratify': True
        },
        'feature_info': {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'feature_types': {col: config['feature_types'].get(col, 'unknown') for col in feature_names}
        },
        'sample_info': {
            'n_train_samples': len(X_train_df),
            'n_test_samples': len(X_test_df),
            'train_class_distribution': train_class_dist,
            'test_class_distribution': test_class_dist
        },
        'data_quality': {
            'train_missing_values': X_train_df.isnull().sum().to_dict(),
            'test_missing_values': X_test_df.isnull().sum().to_dict(),
            'train_duplicates': X_train_df.duplicated().sum(),
            'test_duplicates': X_test_df.duplicated().sum()
        },
        'statistics': {
            'train_numeric_stats': X_train_df.select_dtypes(include=[np.number]).describe().to_dict(),
            'test_numeric_stats': X_test_df.select_dtypes(include=[np.number]).describe().to_dict()
        }
    }
    
    # Save metadata to all format directories
    for format_dir in [csv_dir, npy_dir, pickle_dir]:
        with open(f"{format_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"   💾 Saved split datasets for {dataset_name}")
    print(f"      📁 CSV: {csv_dir}")
    print(f"      📁 NumPy: {npy_dir}")
    print(f"      📁 Pickle: {pickle_dir}")
    
    return metadata

def load_split_datasets(dataset_name, format_type='csv'):
    """
    Load previously saved split datasets
    
    Args:
        dataset_name: Name of the dataset
        format_type: 'csv', 'npy', or 'pickle'
    
    Returns:
        X_train, X_test, y_train, y_test, metadata
    """
    
    base_dir = f"split_datasets/{format_type}/{dataset_name}"
    
    if format_type == 'csv':
        X_train = pd.read_csv(f"{base_dir}/X_train.csv")
        X_test = pd.read_csv(f"{base_dir}/X_test.csv")
        y_train = pd.read_csv(f"{base_dir}/y_train.csv")['target']
        y_test = pd.read_csv(f"{base_dir}/y_test.csv")['target']
        
    elif format_type == 'npy':
        X_train = np.load(f"{base_dir}/X_train.npy")
        X_test = np.load(f"{base_dir}/X_test.npy")
        y_train = np.load(f"{base_dir}/y_train.npy")
        y_test = np.load(f"{base_dir}/y_test.npy")
        
    elif format_type == 'pickle':
        with open(f"{base_dir}/X_train.pkl", 'rb') as f:
            X_train = pickle.load(f)
        with open(f"{base_dir}/X_test.pkl", 'rb') as f:
            X_test = pickle.load(f)
        with open(f"{base_dir}/y_train.pkl", 'rb') as f:
            y_train = pickle.load(f)
        with open(f"{base_dir}/y_test.pkl", 'rb') as f:
            y_test = pickle.load(f)
    
    else:
        raise ValueError("format_type must be 'csv', 'npy', or 'pickle'")
    
    # Load metadata
    with open(f"{base_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return X_train, X_test, y_train, y_test, metadata

def preprocess_dataset(dataset_name, config):
    """Enhanced dataset preprocessing with comprehensive error handling"""
    
    print(f"   📊 Loading and preprocessing {config['name']}...")
    
    try:
        # Load data
        data = pd.read_csv(config['path'])
        print(f"   📈 Loaded {len(data)} samples with {len(data.columns)} columns")
        
        # Drop specified columns
        if 'drop_columns' in config:
            existing_drop_cols = [col for col in config['drop_columns'] if col in data.columns]
            if existing_drop_cols:
                data = data.drop(columns=existing_drop_cols)
                print(f"   🗑️  Dropped columns: {existing_drop_cols}")
        
        # Handle missing values intelligently
        initial_missing = data.isnull().sum().sum()
        if initial_missing > 0:
            print(f"   🔧 Handling {initial_missing} missing values...")
            
            # Numeric columns: fill with median
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            
            # Categorical columns: fill with mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Handle target variable
        target_col = config['target_column']
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Convert target to numeric if needed
        if data[target_col].dtype == 'object':
            class_mapping = {v: k for k, v in config['class_labels'].items()}
            data[target_col] = data[target_col].map(class_mapping)
            print(f"   🔄 Converted target labels using mapping: {class_mapping}")
        
        # Remove rows with missing target
        before_dropna = len(data)
        data = data.dropna(subset=[target_col])
        after_dropna = len(data)
        if before_dropna != after_dropna:
            print(f"   🧹 Removed {before_dropna - after_dropna} rows with missing target")
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        
        print(f"   ✅ Preprocessing complete")
        print(f"      Train: {len(X_train)} samples")
        print(f"      Test: {len(X_test)} samples")
        print(f"      Features: {len(X.columns)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"   ❌ Error in preprocessing: {str(e)}")
        raise

def get_feature_types(config, X):
    """Determine numeric and categorical features"""
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        if col in config['feature_types']:
            ftype = config['feature_types'][col]
            if ftype == 'numeric':
                numeric_features.append(col)
            elif ftype in ['categorical', 'binary']:
                categorical_features.append(col)
        else:
            # Infer type from data
            if X[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)
    
    return numeric_features, categorical_features

def get_enhanced_param_grids():
    """Enhanced parameter grids for comprehensive hyperparameter tuning"""
    return {
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 7, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__class_weight': [None, 'balanced']
        },
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': [None, 'balanced']
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0]
        },
        'MLP': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50), (200, 100)],
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
            'classifier__learning_rate': ['constant', 'adaptive'],
            'classifier__solver': ['adam', 'lbfgs']
        }
    }

def build_enhanced_model_pipelines(numeric_features, categorical_features):
    """Build comprehensive model pipelines with advanced preprocessing"""
    
    # Enhanced preprocessors
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    scale_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Enhanced model pipelines
    models = {
        'Decision Tree': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', DecisionTreeClassifier(random_state=SEED))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', LogisticRegression(max_iter=2000, random_state=SEED))
        ]),
        'MLP': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', MLPClassifier(
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=SEED,
                max_iter=2000
            ))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=SEED,
                n_jobs=-1,
                oob_score=True
            ))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=SEED,
                n_jobs=-1
            ))
        ])
    }
    
    return models

def tune_hyperparameters(pipeline, param_grid, X_train, y_train, cv=5):
    """Enhanced hyperparameter tuning with smart search strategy"""
    
    # Calculate total parameter combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    # Use RandomizedSearchCV for large parameter spaces
    if total_combinations > 100:
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=min(100, total_combinations),
            cv=cv,
            scoring='f1_weighted',
            random_state=SEED,
            n_jobs=-1,
            verbose=0
        )
        search_type = "RandomizedSearchCV"
    else:
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        search_type = "GridSearchCV"
    
    print(f"      Using {search_type} with {total_combinations} combinations")
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_, search.best_score_

def evaluate_model_comprehensive(model, X_test, y_test, class_labels):
    """Comprehensive model evaluation with detailed metrics"""
    
    y_pred = model.predict(X_test)
    
    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2:  # Binary classification
            y_proba_pos = y_proba[:, 1]
        else:  # Multi-class
            y_proba_pos = None
    else:
        y_proba_pos = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_micro': recall_score(y_test, y_pred, average='micro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_micro': f1_score(y_test, y_pred, average='micro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Add ROC AUC if possible
    if y_proba_pos is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba_pos)
        except:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics

def save_model_with_metadata(model, model_name, dataset_name, metrics, best_params, config):
    """Save model with comprehensive metadata"""
    
    # Create model filename
    model_filename = f"{dataset_name}_{model_name.replace(' ', '_')}_tuned.pkl"
    model_path = f"saved_models/{model_filename}"
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Create model metadata
    model_metadata = {
        'model_info': {
            'name': model_name,
            'filename': model_filename,
            'dataset': dataset_name,
            'dataset_name': config['name'],
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'random_state': SEED
        },
        'hyperparameters': {
            'best_params': best_params,
            'tuning_method': 'GridSearchCV/RandomizedSearchCV'
        },
        'performance': metrics,
        'dataset_info': {
            'target_column': config['target_column'],
            'class_labels': config['class_labels']
        }
    }
    
    # Save metadata
    metadata_path = f"saved_models/{dataset_name}_{model_name.replace(' ', '_')}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    return model_path, metadata_path

def main():
    """Main execution function"""
    
    print("🚀 Starting Enhanced ML Pipeline")
    print("=" * 60)
    
    # Create output directories
    create_output_directories()
    
    # Initialize results storage
    full_results = []
    
    # Process each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"🎯 Processing Dataset: {config['name']} ({dataset_name})")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocess dataset
            X_train, X_test, y_train, y_test = preprocess_dataset(dataset_name, config)
            
            # Step 2: Save split datasets
            metadata = save_split_datasets(dataset_name, X_train, X_test, y_train, y_test, config)
            
            # Step 3: Prepare for model training
            numeric_features, categorical_features = get_feature_types(config, X_train)
            
            print(f"\n   📋 Dataset Summary:")
            print(f"      Total Features: {len(X_train.columns)}")
            print(f"      Numeric Features: {len(numeric_features)}")
            print(f"      Categorical Features: {len(categorical_features)}")
            print(f"      Training Samples: {len(X_train)}")
            print(f"      Test Samples: {len(X_test)}")
            print(f"      Class Distribution: {dict(y_train.value_counts().sort_index())}")
            
            # Step 4: Build models and parameter grids
            models = build_enhanced_model_pipelines(numeric_features, categorical_features)
            param_grids = get_enhanced_param_grids()
            dataset_results = []
            
            # Step 5: Train and evaluate each model
            for model_name, pipeline in models.items():
                print(f"\n   🤖 Training {model_name}...")
                
                try:
                    # Hyperparameter tuning
                    if model_name in param_grids:
                        best_model, best_params, best_cv_score = tune_hyperparameters(
                            pipeline, param_grids[model_name], X_train, y_train
                        )
                        print(f"      Best CV F1 Score: {best_cv_score:.4f}")
                        print(f"      Best Parameters: {best_params}")
                    else:
                        best_model = pipeline
                        best_model.fit(X_train, y_train)
                        best_cv_score = None
                        best_params = {}
                    
                    # Evaluate on test set
                    metrics = evaluate_model_comprehensive(best_model, X_test, y_test, config['class_labels'])
                    
                    # Save model with metadata
                    model_path, metadata_path = save_model_with_metadata(
                        best_model, model_name, dataset_name, metrics, best_params, config
                    )
                    
                    # Store results
                    result_record = {
                        'dataset': dataset_name,
                        'dataset_name': config['name'],
                        'model': model_name,
                        'accuracy': metrics['accuracy'],
                        'precision_weighted': metrics['precision_weighted'],
                        'recall_weighted': metrics['recall_weighted'],
                        'f1_weighted': metrics['f1_weighted'],
                        'roc_auc': metrics['roc_auc'],
                        'cv_f1_score': best_cv_score,
                        'model_path': model_path,
                        'metadata_path': metadata_path
                    }
                    
                    dataset_results.append(result_record)
                    full_results.append(result_record)
                
                    # Save model
                    model_path = f"saved_models/{dataset_name}_{model_name.replace(' ', '_')}_tuned.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model, f)
                    
                    print(f"✅ Trained & saved {model_name}")
                    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
                    print(f"   Test F1: {metrics['f1_weighted']:.4f}")
                    print(f"   Test AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "   Test AUC: N/A")
                
                except Exception as e:
                    print(f"❌ Error training {model_name}: {str(e)}")
                    continue
        except Exception as e:
            print(f"❌ Error processing dataset {dataset_name}: {str(e)}")
            continue

    # Generate final comprehensive report
    if full_results:
        final_report = pd.DataFrame(full_results)
        final_report.to_csv("reports/full_tuned_evaluation_report.csv", index=False)
        
        print(f"\n{'='*50}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*50}")
        
        # Summary by dataset
        for dataset in final_report['dataset'].unique():
            dataset_data = final_report[final_report['dataset'] == dataset]
            best_model = dataset_data.loc[dataset_data['f1'].idxmax()]
            print(f"\n{dataset.upper()}:")
            print(f"  Best Model: {best_model['model']}")
            print(f"  F1 Score: {best_model['f1']:.4f}")
            print(f"  Accuracy: {best_model['accuracy']:.4f}")
            print(f"  ROC AUC: {best_model['roc_auc']:.4f}" if best_model['roc_auc'] else "  ROC AUC: N/A")
        
        # Overall best performing models
        print(f"\n{'='*30}")
        print("TOP 5 MODELS OVERALL (by F1):")
        print(f"{'='*30}")
        top_models = final_report.nlargest(5, 'f1')
        for idx, row in top_models.iterrows():
            print(f"{row['dataset']} - {row['model']}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}")
        
        print(f"\nFull tuned evaluation report saved to reports/full_tuned_evaluation_report.csv")
    else:
        print("\nNo results generated due to errors")

    print("\nEnhanced processing complete!")

if __name__ == "__main__":
    main()