import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
try:
    from config import DATASETS, ML_MODELS
except ImportError:
    print("❌ Error: Could not import config module. Please ensure config.py exists.")
    sys.exit(1)

# Constants
SEED = 42
np.random.seed(SEED)
CATEGORICAL_THRESHOLD = 15  # Max unique values for one-hot encoding

# Create output directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("preprocessed_data", exist_ok=True)

def encode_categorical(X_train, X_test, categorical_features):
    """Encodes categorical features using appropriate encoding strategies"""
    # Create copies to avoid modifying original data
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    encoders = {}
    
    for feature in categorical_features:
        # Handle high cardinality features with frequency encoding
        if X_train[feature].nunique() > CATEGORICAL_THRESHOLD:
            freq_map = X_train[feature].value_counts(normalize=True).to_dict()
            X_train_encoded[feature] = X_train[feature].map(freq_map)
            X_test_encoded[feature] = X_test[feature].map(freq_map).fillna(0)  # Handle unseen categories
            encoders[feature] = {'type': 'frequency', 'mapping': freq_map}
            
        # Handle binary features with label encoding
        elif X_train[feature].nunique() == 2:
            le = LabelEncoder()
            X_train_encoded[feature] = le.fit_transform(X_train[feature])
            X_test_encoded[feature] = le.transform(X_test[feature])
            encoders[feature] = {'type': 'label', 'mapping': dict(zip(le.classes_, le.transform(le.classes_)))}
            
        # Handle other categorical features with one-hot encoding
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # Fit on training data
            train_encoded = ohe.fit_transform(X_train[[feature]])
            test_encoded = ohe.transform(X_test[[feature]])
            
            # Create feature names
            feature_names = [f"{feature}_{cat}" for cat in ohe.categories_[0]]
            
            # Create DataFrames for encoded features
            train_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
            test_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)
            
            # Replace original feature with encoded features
            X_train_encoded = pd.concat([X_train_encoded.drop(feature, axis=1), train_df], axis=1)
            X_test_encoded = pd.concat([X_test_encoded.drop(feature, axis=1), test_df], axis=1)
            
            encoders[feature] = {'type': 'onehot', 'categories': ohe.categories_[0].tolist()}
    
    return X_train_encoded, X_test_encoded, encoders

def preprocess_dataset(dataset_name, config):
    """Load and preprocess dataset with categorical encoding"""
    # Load data
    try:
        data = pd.read_csv(config['path'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {config['path']}")
    
    # Drop specified columns
    if 'drop_columns' in config:
        data = data.drop(columns=[col for col in config['drop_columns'] if col in data.columns])
    
    # Handle missing values
    # For numeric columns: fill with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # For categorical columns: fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_values = data[col].mode()
            if len(mode_values) > 0:
                data[col].fillna(mode_values[0], inplace=True)
            else:
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
    
    # Check data availability
    if len(data) == 0:
        raise ValueError(f"No valid data remaining after preprocessing for {dataset_name}")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split data with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Identify feature types
    numeric_features, categorical_features = get_feature_types(config, X_train)
    
    # Encode categorical features
    encoding_info = {}
    if categorical_features:
        X_train, X_test, encoding_info = encode_categorical(X_train, X_test, categorical_features)
    
    # Convert all data to numeric (final safety check)
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_train[col].fillna(0, inplace=True)
        X_test[col].fillna(0, inplace=True)
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features, encoding_info

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

def save_preprocessed_data(dataset_name, config, X_train, X_test, y_train, y_test, 
                          numeric_features, categorical_features, encoding_info):
    """Save preprocessed datasets and metadata"""
    dataset_dir = f"preprocessed_data/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(f"{dataset_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{dataset_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{dataset_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{dataset_dir}/y_test.csv", index=False)
    
    # Convert numpy types to Python native types
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj

    # Prepare metadata with type conversion
    metadata = {
        "dataset_name": config["name"],
        "target_column": config["target_column"],
        "class_labels": config.get("class_labels", {}),
        "original_features": {
            "numeric": numeric_features,
            "categorical": categorical_features
        },
        "final_features": list(X_train.columns),
        "encoding_info": convert_to_python_types(encoding_info),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": {
            "train": convert_to_python_types(y_train.value_counts().to_dict()),
            "test": convert_to_python_types(y_test.value_counts().to_dict())
        }
    }
    
    # Save metadata with explicit type handling
    with open(f"{dataset_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)  # Use default=str for safety
    
    print(f"💾 Saved preprocessed data to {dataset_dir}")
    return metadata

if __name__ == "__main__":
    for ds_name, ds_cfg in DATASETS.items():
        try:
            print(f"\n▶️ Preprocessing {ds_name} dataset...")
            X_train, X_test, y_train, y_test, num_feats, cat_feats, enc_info = preprocess_dataset(ds_name, ds_cfg)
            metadata = save_preprocessed_data(
                ds_name, ds_cfg, 
                X_train, X_test, y_train, y_test,
                num_feats, cat_feats, enc_info
            )
            
            # Print dataset summary
            print(f"✅ Preprocessing complete")
            print(f"   Original features: {len(num_feats + cat_feats)}")
            print(f"   Final features: {len(metadata['final_features'])}")
            print(f"   Train size: {metadata['train_size']}")
            print(f"   Test size: {metadata['test_size']}")
            print(f"   Class distribution (train): {metadata['class_distribution']['train']}")
            
        except Exception as e:
            print(f"❌ Error preprocessing {ds_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All datasets preprocessed and saved")