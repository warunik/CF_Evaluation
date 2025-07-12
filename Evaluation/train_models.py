import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
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

SEED = 42
np.random.seed(SEED)

# Configuration from your provided data
from config import DATASETS, ML_MODELS

# Create output directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Initialize results storage
full_results = []

def preprocess_dataset(dataset_name, config):
    """Load and preprocess dataset with enhanced preprocessing"""
    # Load data
    data = pd.read_csv(config['path'])
    
    # Drop specified columns
    if 'drop_columns' in config:
        data = data.drop(columns=[col for col in config['drop_columns'] if col in data.columns])
    
    # Handle missing values more intelligently
    # For numeric columns, fill with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Handle target variable
    target_col = config['target_column']
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Convert target to numeric using class labels if needed
    if data[target_col].dtype == 'object':
        class_mapping = {v: k for k, v in config['class_labels'].items()}
        data[target_col] = data[target_col].map(class_mapping)
    
    # Remove rows with missing target
    data = data.dropna(subset=[target_col])
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split data with stratification
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

def get_feature_types(config, X):
    """Get numeric and categorical features based on config and data types"""
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

def get_param_grids():
    """Define parameter grids for hyperparameter tuning"""
    return {
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 7, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        },
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 0.9, 1.0]
        },
        'MLP': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }
    }

def build_enhanced_model_pipelines(numeric_features, categorical_features):
    """Create enhanced model pipelines with feature selection and better preprocessing"""
    
    # Enhanced preprocessors
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),  # RobustScaler handles outliers better
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    scale_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Model definitions with better default parameters
    models = {
        'Decision Tree': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', DecisionTreeClassifier(random_state=SEED))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))
        ]),
        'MLP': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', MLPClassifier(
                early_stopping=True, 
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=SEED,
                max_iter=1000
            ))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=SEED,
                n_jobs=-1
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
    """Perform hyperparameter tuning using GridSearchCV"""
    # Use RandomizedSearchCV for larger parameter spaces
    if len(param_grid) > 20:
        search = RandomizedSearchCV(
            pipeline, 
            param_grid, 
            n_iter=50,
            cv=cv, 
            scoring='f1',
            random_state=SEED,
            n_jobs=-1,
            verbose=0
        )
    else:
        search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

# Main processing loop with enhanced tuning
for dataset_name, config in DATASETS.items():
    try:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Preprocess dataset
        X_train, X_test, y_train, y_test = preprocess_dataset(dataset_name, config)
        numeric_features, categorical_features = get_feature_types(config, X_train)
        
        print(f"Features: {len(X_train.columns)} (Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)})")
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"Class distribution: {dict(y_train.value_counts())}")
        
        # Build models
        models = build_enhanced_model_pipelines(numeric_features, categorical_features)
        param_grids = get_param_grids()
        dataset_results = []
        
        for model_name, pipeline in models.items():
            print(f"\nTraining and tuning {model_name}...")
            
            try:
                # Hyperparameter tuning
                if model_name in param_grids:
                    best_model, best_params, best_cv_score = tune_hyperparameters(
                        pipeline, param_grids[model_name], X_train, y_train
                    )
                    print(f"   Best CV F1 Score: {best_cv_score:.4f}")
                    print(f"   Best Parameters: {best_params}")
                else:
                    # Just train the model without tuning
                    best_model = pipeline
                    best_model.fit(X_train, y_train)
                    best_cv_score = None
                    best_params = None
                
                # Evaluate on test set
                metrics = evaluate_model(best_model, X_test, y_test)
                
                # Save results
                model_metrics = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc'],
                    'cv_f1_score': best_cv_score,
                    'best_params': str(best_params) if best_params else None
                }
                dataset_results.append(model_metrics)
                full_results.append(model_metrics)
                
                # Save model
                model_path = f"saved_models/{dataset_name}_{model_name.replace(' ', '_')}_tuned.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                
                print(f"✅ Trained & saved {model_name}")
                print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Test F1: {metrics['f1']:.4f}")
                print(f"   Test AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "   Test AUC: N/A")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Save dataset report
        if dataset_results:
            report_df = pd.DataFrame(dataset_results)
            report_df.to_csv(f"reports/{dataset_name}_tuned_report.csv", index=False)
            
            # Print best model for this dataset
            best_model_row = report_df.loc[report_df['f1'].idxmax()]
            print(f"\n🏆 Best model for {dataset_name}: {best_model_row['model']}")
            print(f"   F1 Score: {best_model_row['f1']:.4f}")
            print(f"   Accuracy: {best_model_row['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")

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