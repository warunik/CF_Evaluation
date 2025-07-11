import os
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
    "compas": {
        "name": "Recidivism Prediction",
        "path": "Evaluation/data/compas.csv",
        "target_column": "is_recid",
        "drop_columns": [
            'id', 'name', 'first', 'last', 
            'compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out',
            'c_case_number', 'c_offense_date', 'c_arrest_date',
            'r_case_number', 'r_charge_degree', 'r_offense_date', 'r_charge_desc',
            'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_offense_date',
            'vr_charge_desc', 'type_of_assessment', 'decile_score.1',
            'score_text', 'screening_date', 'v_type_of_assessment',
            'v_decile_score', 'v_score_text', 'v_screening_date',
            'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
            'event', 'decile_score'
        ],
        "class_labels": {0: "No Recidivism", 1: "Recidivism"},
        "feature_types": {
            "sex": "categorical",
            "age": "numeric",
            "age_cat": "categorical",
            "race": "categorical",
            "juv_fel_count": "numeric",
            "juv_misd_count": "numeric",
            "juv_other_count": "numeric",
            "priors_count": "numeric",
            "days_b_screening_arrest": "numeric",
            "c_days_from_compas": "numeric",
            "c_charge_degree": "categorical",
            "c_charge_desc": "categorical",
            "r_days_from_arrest": "numeric",
            "violent_recid": "numeric",
            "is_violent_recid": "numeric",
            "vr_charge_degree": "categorical",
            "two_year_recid": "numeric"
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