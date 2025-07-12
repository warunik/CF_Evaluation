import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Configuration from your provided data
try:
    from config import DATASETS, ML_MODELS
except ImportError:
    print("❌ Error: Could not import config module. Please ensure config.py exists.")
    sys.exit(1)

# Create output directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("preprocessed_data", exist_ok=True)


def preprocess_dataset(dataset_name, config):
    """Load and preprocess dataset with enhanced preprocessing"""
    # Load data
    try:
        data = pd.read_csv(config['path'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {config['path']}")
    
    # Drop specified columns
    if 'drop_columns' in config:
        data = data.drop(columns=[col for col in config['drop_columns'] if col in data.columns])
    
    # Handle missing values more intelligently
    # For numeric columns, fill with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # For categorical columns, fill with mode (with safety check)
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_values = data[col].mode()
            if len(mode_values) > 0:
                data[col].fillna(mode_values[0], inplace=True)
            else:
                # If no mode (all NaN), fill with 'Unknown'
                data[col].fillna('Unknown', inplace=True)
    
    # Handle target variable
    target_col = config['target_column']
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Convert target to numeric using class labels if needed
    if data[target_col].dtype == 'object' and 'class_labels' in config:
        class_mapping = {v: k for k, v in config['class_labels'].items()}
        data[target_col] = data[target_col].map(class_mapping)
    
    # Remove rows with missing target
    data = data.dropna(subset=[target_col])
    
    # Check if we have enough data
    if len(data) == 0:
        raise ValueError(f"No valid data remaining after preprocessing for {dataset_name}")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split data with stratification (with fallback)
    try:
        return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    except ValueError as e:
        # If stratification fails, split without stratification
        print(f"⚠️  Warning: Stratification failed for {dataset_name}, using random split: {e}")
        return train_test_split(X, y, test_size=0.3, random_state=SEED)

def get_feature_types(config, X):
    """Get numeric and categorical features based on config and data types"""
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        if 'feature_types' in config and col in config['feature_types']:
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

def save_preprocessed_data(dataset_name, config, X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    """Save preprocessed datasets and metadata for future use"""
    dataset_dir = f"preprocessed_data/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(f"{dataset_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{dataset_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{dataset_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{dataset_dir}/y_test.csv", index=False)
    
    # Prepare metadata
    metadata = {
        "dataset_name": config["name"],
        "target_column": config["target_column"],
        "class_labels": config.get("class_labels", {}),  # Safe access
        "feature_types": {
            col: ("numeric" if col in numeric_features else "categorical")
            for col in X_train.columns
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": list(X_train.columns),
        "class_distribution": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().items()}
        }
    }
    
    # Save metadata
    with open(f"{dataset_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"💾 Saved preprocessed data to {dataset_dir}")

if __name__ == "__main__":
    
    for ds_name, ds_cfg in DATASETS.items():
        try:
            print(f"▶️ Preprocessing {ds_name}…")
            X_train, X_test, y_train, y_test = preprocess_dataset(ds_name, ds_cfg)
            num_feats, cat_feats = get_feature_types(ds_cfg, X_train)
            save_preprocessed_data(ds_name, ds_cfg, X_train, X_test, y_train, y_test, num_feats, cat_feats)
        except Exception as e:
            print(f"❌ Error preprocessing {ds_name}: {e}")
            continue
    
    print("✅ All datasets preprocessed.")