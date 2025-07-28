# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, 
#     roc_auc_score, confusion_matrix, classification_report
# )
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.feature_selection import SelectKBest, f_classif
# import warnings
# warnings.filterwarnings('ignore')

# SEED = 42
# np.random.seed(SEED)

# # Configuration from your provided data
# from config import DATASETS, ML_MODELS

# # Create output directories
# os.makedirs("saved_models", exist_ok=True)
# os.makedirs("reports", exist_ok=True)

# # Initialize results storage
# full_results = []

# def preprocess_dataset(dataset_name, config):
#     """Load and preprocess dataset with enhanced preprocessing"""
#     # Load data
#     data = pd.read_csv(config['path'])
    
#     # Drop specified columns
#     if 'drop_columns' in config:
#         data = data.drop(columns=[col for col in config['drop_columns'] if col in data.columns])
    
#     # Handle missing values more intelligently
#     # For numeric columns, fill with median
#     numeric_cols = data.select_dtypes(include=[np.number]).columns
#     for col in numeric_cols:
#         if data[col].isnull().sum() > 0:
#             data[col].fillna(data[col].median(), inplace=True)
    
#     # For categorical columns, fill with mode
#     categorical_cols = data.select_dtypes(include=['object']).columns
#     for col in categorical_cols:
#         if data[col].isnull().sum() > 0:
#             data[col].fillna(data[col].mode()[0], inplace=True)
    
#     # Handle target variable
#     target_col = config['target_column']
#     if target_col not in data.columns:
#         raise ValueError(f"Target column '{target_col}' not found in dataset")
    
#     # Convert target to numeric using class labels if needed
#     if data[target_col].dtype == 'object':
#         class_mapping = {v: k for k, v in config['class_labels'].items()}
#         data[target_col] = data[target_col].map(class_mapping)
    
#     # Remove rows with missing target
#     data = data.dropna(subset=[target_col])
    
#     # Separate features and target
#     X = data.drop(columns=[target_col])
#     y = data[target_col]
    
#     # Split data with stratification
#     return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

# def get_feature_types(config, X):
#     """Get numeric and categorical features based on config and data types"""
#     numeric_features = []
#     categorical_features = []
    
#     for col in X.columns:
#         if col in config['feature_types']:
#             ftype = config['feature_types'][col]
#             if ftype == 'numeric':
#                 numeric_features.append(col)
#             elif ftype in ['categorical', 'binary']:
#                 categorical_features.append(col)
#         else:
#             # Infer type from data
#             if X[col].dtype in ['int64', 'float64']:
#                 numeric_features.append(col)
#             else:
#                 categorical_features.append(col)
    
#     return numeric_features, categorical_features

# def get_param_grids():
#     """Define parameter grids for hyperparameter tuning"""
#     return {
#         'Decision Tree': {
#             'classifier__max_depth': [3, 5, 7, 10, None],
#             'classifier__min_samples_split': [2, 5, 10],
#             'classifier__min_samples_leaf': [1, 2, 4],
#             'classifier__criterion': ['gini', 'entropy']
#         },
#         'Logistic Regression': {
#             'classifier__C': [0.01, 0.1, 1, 10, 100],
#             'classifier__penalty': ['l1', 'l2'],
#             'classifier__solver': ['liblinear', 'saga']
#         },
#         'Random Forest': {
#             'classifier__n_estimators': [50, 100, 200],
#             'classifier__max_depth': [5, 10, 15, None],
#             'classifier__min_samples_split': [2, 5, 10],
#             'classifier__min_samples_leaf': [1, 2, 4]
#         },
#         'XGBoost': {
#             'classifier__n_estimators': [50, 100, 200],
#             'classifier__learning_rate': [0.01, 0.1, 0.2],
#             'classifier__max_depth': [3, 5, 7],
#             'classifier__subsample': [0.8, 0.9, 1.0]
#         },
#         'MLP': {
#             'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
#             'classifier__alpha': [0.0001, 0.001, 0.01],
#             'classifier__learning_rate': ['constant', 'adaptive']
#         }
#     }

# def build_enhanced_model_pipelines(numeric_features, categorical_features):
#     """Create enhanced model pipelines with feature selection and better preprocessing"""
    
#     # Enhanced preprocessors
#     tree_preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', RobustScaler(), numeric_features),  # RobustScaler handles outliers better
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
#         ])
    
#     scale_preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
#         ])
    
#     # Model definitions with better default parameters
#     models = {
#         'Decision Tree': Pipeline([
#             ('preprocessor', tree_preprocessor),
#             ('feature_selection', SelectKBest(f_classif, k='all')),
#             ('classifier', DecisionTreeClassifier(random_state=SEED))
#         ]),
#         'Logistic Regression': Pipeline([
#             ('preprocessor', scale_preprocessor),
#             ('feature_selection', SelectKBest(f_classif, k='all')),
#             ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))
#         ]),
#         'MLP': Pipeline([
#             ('preprocessor', scale_preprocessor),
#             ('feature_selection', SelectKBest(f_classif, k='all')),
#             ('classifier', MLPClassifier(
#                 early_stopping=True, 
#                 validation_fraction=0.1,
#                 n_iter_no_change=10,
#                 random_state=SEED,
#                 max_iter=1000
#             ))
#         ]),
#         'Random Forest': Pipeline([
#             ('preprocessor', tree_preprocessor),
#             ('classifier', RandomForestClassifier(
#                 n_estimators=100,
#                 random_state=SEED,
#                 n_jobs=-1
#             ))
#         ]),
#         'XGBoost': Pipeline([
#             ('preprocessor', tree_preprocessor),
#             ('classifier', XGBClassifier(
#                 objective='binary:logistic',
#                 use_label_encoder=False,
#                 eval_metric='logloss',
#                 random_state=SEED,
#                 n_jobs=-1
#             ))
#         ])
#     }
    
#     return models

# def tune_hyperparameters(pipeline, param_grid, X_train, y_train, cv=5):
#     """Perform hyperparameter tuning using GridSearchCV"""
#     # Use RandomizedSearchCV for larger parameter spaces
#     if len(param_grid) > 20:
#         search = RandomizedSearchCV(
#             pipeline, 
#             param_grid, 
#             n_iter=50,
#             cv=cv, 
#             scoring='f1',
#             random_state=SEED,
#             n_jobs=-1,
#             verbose=0
#         )
#     else:
#         search = GridSearchCV(
#             pipeline, 
#             param_grid, 
#             cv=cv, 
#             scoring='f1',
#             n_jobs=-1,
#             verbose=0
#         )
    
#     search.fit(X_train, y_train)
#     return search.best_estimator_, search.best_params_, search.best_score_

# def evaluate_model(model, X_test, y_test):
#     """Evaluate model and return comprehensive metrics"""
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
#     return {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, average='weighted'),
#         'recall': recall_score(y_test, y_pred, average='weighted'),
#         'f1': f1_score(y_test, y_pred, average='weighted'),
#         'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
#         'confusion_matrix': confusion_matrix(y_test, y_pred),
#         'classification_report': classification_report(y_test, y_pred, output_dict=True)
#     }

# # Main processing loop with enhanced tuning
# for dataset_name, config in DATASETS.items():
#     try:
#         print(f"\n{'='*50}")
#         print(f"Processing dataset: {dataset_name}")
#         print(f"{'='*50}")
        
#         # Preprocess dataset
#         X_train, X_test, y_train, y_test = preprocess_dataset(dataset_name, config)
#         numeric_features, categorical_features = get_feature_types(config, X_train)
        
#         print(f"Features: {len(X_train.columns)} (Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)})")
#         print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
#         print(f"Class distribution: {dict(y_train.value_counts())}")
        
#         # Build models
#         models = build_enhanced_model_pipelines(numeric_features, categorical_features)
#         param_grids = get_param_grids()
#         dataset_results = []
        
#         for model_name, pipeline in models.items():
#             print(f"\nTraining and tuning {model_name}...")
            
#             try:
#                 # Hyperparameter tuning
#                 if model_name in param_grids:
#                     best_model, best_params, best_cv_score = tune_hyperparameters(
#                         pipeline, param_grids[model_name], X_train, y_train
#                     )
#                     print(f"   Best CV F1 Score: {best_cv_score:.4f}")
#                     print(f"   Best Parameters: {best_params}")
#                 else:
#                     # Just train the model without tuning
#                     best_model = pipeline
#                     best_model.fit(X_train, y_train)
#                     best_cv_score = None
#                     best_params = None
                
#                 # Evaluate on test set
#                 metrics = evaluate_model(best_model, X_test, y_test)
                
#                 # Save results
#                 model_metrics = {
#                     'dataset': dataset_name,
#                     'model': model_name,
#                     'accuracy': metrics['accuracy'],
#                     'precision': metrics['precision'],
#                     'recall': metrics['recall'],
#                     'f1': metrics['f1'],
#                     'roc_auc': metrics['roc_auc'],
#                     'cv_f1_score': best_cv_score,
#                     'best_params': str(best_params) if best_params else None
#                 }
#                 dataset_results.append(model_metrics)
#                 full_results.append(model_metrics)
                
#                 # Save model
#                 model_path = f"saved_models/{dataset_name}_{model_name.replace(' ', '_')}_tuned.pkl"
#                 with open(model_path, 'wb') as f:
#                     pickle.dump(best_model, f)
                
#                 print(f"✅ Trained & saved {model_name}")
#                 print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
#                 print(f"   Test F1: {metrics['f1']:.4f}")
#                 print(f"   Test AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "   Test AUC: N/A")
                
#             except Exception as e:
#                 print(f"❌ Error training {model_name}: {str(e)}")
#                 continue
        
#         # Save dataset report
#         if dataset_results:
#             report_df = pd.DataFrame(dataset_results)
#             report_df.to_csv(f"reports/{dataset_name}_tuned_report.csv", index=False)
            
#             # Print best model for this dataset
#             best_model_row = report_df.loc[report_df['f1'].idxmax()]
#             print(f"\n🏆 Best model for {dataset_name}: {best_model_row['model']}")
#             print(f"   F1 Score: {best_model_row['f1']:.4f}")
#             print(f"   Accuracy: {best_model_row['accuracy']:.4f}")
        
#     except Exception as e:
#         print(f"❌ Error processing {dataset_name}: {str(e)}")

# # Generate final comprehensive report
# if full_results:
#     final_report = pd.DataFrame(full_results)
#     final_report.to_csv("reports/full_tuned_evaluation_report.csv", index=False)
    
#     print(f"\n{'='*50}")
#     print("FINAL RESULTS SUMMARY")
#     print(f"{'='*50}")
    
#     # Summary by dataset
#     for dataset in final_report['dataset'].unique():
#         dataset_data = final_report[final_report['dataset'] == dataset]
#         best_model = dataset_data.loc[dataset_data['f1'].idxmax()]
#         print(f"\n{dataset.upper()}:")
#         print(f"  Best Model: {best_model['model']}")
#         print(f"  F1 Score: {best_model['f1']:.4f}")
#         print(f"  Accuracy: {best_model['accuracy']:.4f}")
#         print(f"  ROC AUC: {best_model['roc_auc']:.4f}" if best_model['roc_auc'] else "  ROC AUC: N/A")
    
#     # Overall best performing models
#     print(f"\n{'='*30}")
#     print("TOP 5 MODELS OVERALL (by F1):")
#     print(f"{'='*30}")
#     top_models = final_report.nlargest(5, 'f1')
#     for idx, row in top_models.iterrows():
#         print(f"{row['dataset']} - {row['model']}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}")
    
#     print(f"\nFull tuned evaluation report saved to reports/full_tuned_evaluation_report.csv")
# else:
#     print("\nNo results generated due to errors")

# print("\nEnhanced processing complete!")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold, validation_curve
)
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
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Configuration from your provided data
from config import DATASETS, ML_MODELS

# Create output directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("validation_plots", exist_ok=True)

# Initialize results storage
full_results = []

def preprocess_dataset(dataset_name, config):
    """Load and preprocess dataset with enhanced preprocessing and proper splits"""
    # Load data
    data = pd.read_csv(config['path'])
    
    # Drop specified columns
    if 'drop_columns' in config:
        data = data.drop(columns=[col for col in config['drop_columns'] if col in data.columns])
    
    # Handle missing values more intelligently
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
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
    
    # First split: separate out final test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Second split: train (60%) and validation (20%) from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class EpochBasedMLPClassifier(BaseEstimator, ClassifierMixin):
    """Enhanced MLP with proper epoch-based training and validation monitoring"""
    
    def __init__(self, hidden_layer_sizes=(100,), max_epochs=200, patience=15, 
                 validation_fraction=0.2, alpha=0.001, learning_rate_init=0.001,
                 batch_size='auto', random_state=None, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Training history
        self.train_losses_ = []
        self.val_losses_ = []
        self.train_scores_ = []
        self.val_scores_ = []
        self.best_epoch_ = 0
        
    def fit(self, X, y):
        """Fit with epoch-based training and early stopping"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import log_loss, accuracy_score
        
        # Split training data for validation
        if self.validation_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Initialize the base MLP
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=1,  # We'll control iterations manually
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            random_state=self.random_state,
            warm_start=True,  # Keep training from previous state
            early_stopping=False,  # We handle this manually
            validation_fraction=0.0  # We handle validation manually
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Fit for one iteration
            self.mlp_.max_iter = epoch + 1
            self.mlp_.fit(X_train, y_train)
            
            # Calculate losses and scores
            train_pred_proba = self.mlp_.predict_proba(X_train)
            val_pred_proba = self.mlp_.predict_proba(X_val)
            
            train_loss = log_loss(y_train, train_pred_proba)
            val_loss = log_loss(y_val, val_pred_proba)
            
            train_score = accuracy_score(y_train, self.mlp_.predict(X_train))
            val_score = accuracy_score(y_val, self.mlp_.predict(X_val))
            
            # Store history
            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)
            self.train_scores_.append(train_score)
            self.val_scores_.append(val_score)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch_ = epoch
                patience_counter = 0
                # Save best model state
                self.best_model_state_ = {
                    'coefs_': [coef.copy() for coef in self.mlp_.coefs_],
                    'intercepts_': [intercept.copy() for intercept in self.mlp_.intercepts_]
                }
            else:
                patience_counter += 1
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_score:.4f}, Val Acc={val_score:.4f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (best epoch: {self.best_epoch_})")
                break
        
        # Restore best model state
        if hasattr(self, 'best_model_state_'):
            self.mlp_.coefs_ = self.best_model_state_['coefs_']
            self.mlp_.intercepts_ = self.best_model_state_['intercepts_']
        
        # Store classes for compatibility
        self.classes_ = self.mlp_.classes_
        return self
    
    def predict(self, X):
        return self.mlp_.predict(X)
    
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(len(self.train_losses_))
        
        # Loss plot
        ax1.plot(epochs, self.train_losses_, label='Training Loss', color='blue')
        ax1.plot(epochs, self.val_losses_, label='Validation Loss', color='red')
        ax1.axvline(x=self.best_epoch_, color='green', linestyle='--', label=f'Best Epoch ({self.best_epoch_})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_scores_, label='Training Accuracy', color='blue')
        ax2.plot(epochs, self.val_scores_, label='Validation Accuracy', color='red')
        ax2.axvline(x=self.best_epoch_, color='green', linestyle='--', label=f'Best Epoch ({self.best_epoch_})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class EpochBasedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Enhanced XGBoost with proper epoch-based training and validation monitoring"""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, 
                 early_stopping_rounds=15, validation_fraction=0.2, 
                 random_state=None, verbose=False, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Training history
        self.evals_result_ = {}
        
    def fit(self, X, y):
        """Fit with proper validation and early stopping"""
        from sklearn.model_selection import train_test_split
        
        # Split for validation if needed
        if self.validation_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self.random_state, stratify=y
            )
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
        else:
            X_train, y_train = X, y
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        # Initialize XGBoost with validation
        self.xgb_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit with early stopping
        self.xgb_.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=[],
            verbose=self.verbose
        )
        
        # Store evaluation results
        self.evals_result_ = self.xgb_.evals_result()
        self.classes_ = self.xgb_.classes_
        
        return self
    
    def predict(self, X):
        return self.xgb_.predict(X)
    
    def predict_proba(self, X):
        return self.xgb_.predict_proba(X)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.evals_result_:
            print("No training history available")
            return None
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for eval_name, eval_results in self.evals_result_.items():
            for metric_name, metric_values in eval_results.items():
                epochs = range(len(metric_values))
                ax.plot(epochs, metric_values, label=f'{eval_name}_{metric_name}')
        
        ax.set_xlabel('Boosting Round')
        ax.set_ylabel('Metric Value')
        ax.set_title('XGBoost Training History')
        ax.legend()
        ax.grid(True)
        
        if hasattr(self.xgb_, 'best_iteration'):
            ax.axvline(x=self.xgb_.best_iteration, color='red', linestyle='--', 
                      label=f'Best Iteration ({self.xgb_.best_iteration})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
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

def get_conservative_param_grids():
    """Define conservative parameter grids focused on preventing overfitting"""
    return {
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 7],  # Reduced max depth
            'classifier__min_samples_split': [10, 20, 50],  # Increased min samples
            'classifier__min_samples_leaf': [5, 10, 20],  # Increased min leaf samples
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]  # Feature subsampling
        },
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1.0],  # Stronger regularization
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.9],  # For elasticnet
            'classifier__solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200, 300],  # More trees for stability
            'classifier__max_depth': [5, 10, 15],  # Limited depth
            'classifier__min_samples_split': [10, 20, 50],  # Higher splits
            'classifier__min_samples_leaf': [5, 10, 20],  # Higher leaves
            'classifier__max_features': ['sqrt', 'log2'],  # Feature subsampling
            'classifier__bootstrap': [True],  # Always use bootstrap
            'classifier__oob_score': [True]  # Out-of-bag scoring
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
            'classifier__max_depth': [3, 4, 5],  # Shallow trees
            'classifier__subsample': [0.6, 0.8, 0.9],  # Row subsampling
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],  # Column subsampling
            'classifier__reg_alpha': [0, 0.1, 1],  # L1 regularization
            'classifier__reg_lambda': [1, 10, 100],  # L2 regularization
            'classifier__min_child_weight': [1, 3, 5]  # Minimum samples in leaf
        },
        'MLP': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25)],  # Smaller networks
            'classifier__alpha': [0.001, 0.01, 0.1],  # Regularization
            'classifier__learning_rate_init': [0.01, 0.001, 0.0001],  # Learning rates
            'classifier__max_epochs': [100, 200, 300],  # Number of epochs
            'classifier__patience': [10, 15, 20]  # Early stopping patience
        }
    }

def build_robust_model_pipelines(numeric_features, categorical_features, n_features):
    """Create robust model pipelines with proper regularization and feature selection"""
    
    # Determine optimal number of features (prevent curse of dimensionality)
    max_features = min(n_features // 2, 50)  # Use at most 50 features or half of available
    
    # Enhanced preprocessors with outlier handling
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),  # Handles outliers better
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
        ])
    
    scale_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
        ])
    
    # Model definitions with strong regularization defaults
    models = {
        'Decision Tree': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k=max_features)),
            ('classifier', DecisionTreeClassifier(
                max_depth=5,  # Conservative depth
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=SEED,
                max_features='sqrt'
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k=max_features)),
            ('classifier', LogisticRegression(
                C=0.1,  # Strong regularization
                penalty='elasticnet',
                l1_ratio=0.5,
                solver='saga',
                max_iter=2000,
                random_state=SEED
            ))
        ]),
        'MLP': Pipeline([
            ('preprocessor', scale_preprocessor),
            ('feature_selection', SelectKBest(f_classif, k=max_features)),
            ('classifier', EpochBasedMLPClassifier(
                hidden_layer_sizes=(50,),  # Small network
                alpha=0.01,  # Regularization
                max_epochs=200,  # Maximum epochs
                patience=15,  # Early stopping patience
                validation_fraction=0.2,  # Validation split
                learning_rate_init=0.001,  # Learning rate
                random_state=SEED,
                verbose=False
            ))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=SEED,
                n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', EpochBasedXGBClassifier(
                n_estimators=300,  # More estimators with early stopping
                learning_rate=0.05,  # Lower learning rate
                max_depth=4,
                early_stopping_rounds=20,  # Early stopping
                validation_fraction=0.2,  # Validation split
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=10,
                min_child_weight=3,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=SEED,
                n_jobs=-1,
                verbose=False
            ))
        ])
    }
    
    return models

def tune_hyperparameters_robust(pipeline, param_grid, X_train, y_train, X_val, y_val, cv=10):
    """Perform robust hyperparameter tuning with proper validation"""
    
    # Use StratifiedKFold for better cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    
    # Use RandomizedSearchCV for efficiency with more iterations
    search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=30,  # More iterations for better search
        cv=cv_strategy, 
        scoring='f1',
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        return_train_score=True  # To detect overfitting
    )
    
    search.fit(X_train, y_train)
    
    # Get best model and validate on separate validation set
    best_model = search.best_estimator_
    val_score = f1_score(y_val, best_model.predict(X_val), average='weighted')
    
    # Check for overfitting (large gap between train and validation)
    train_scores = search.cv_results_['mean_train_score']
    cv_scores = search.cv_results_['mean_test_score']
    best_idx = search.best_index_
    
    overfitting_gap = train_scores[best_idx] - cv_scores[best_idx]
    
    return best_model, search.best_params_, search.best_score_, val_score, overfitting_gap

def detect_overfitting(model, X_train, y_train, X_val, y_val):
    """Detect overfitting by comparing train and validation performance"""
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    overfitting_gap = train_f1 - val_f1
    
    # Flag as overfitting if gap > 0.1 (10%)
    is_overfitting = overfitting_gap > 0.1
    
    return {
        'train_f1': train_f1,
        'val_f1': val_f1,
        'overfitting_gap': overfitting_gap,
        'is_overfitting': is_overfitting
    }

def evaluate_model_comprehensive(model, X_test, y_test, X_train, y_train, X_val, y_val):
    """Evaluate model comprehensively including overfitting detection"""
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Overfitting detection
    overfitting_info = detect_overfitting(model, X_train, y_train, X_val, y_val)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        **overfitting_info
    }

# Main processing loop with robust overfitting prevention
for dataset_name, config in DATASETS.items():
    try:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Preprocess dataset with proper train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset(dataset_name, config)
        numeric_features, categorical_features = get_feature_types(config, X_train)
        
        print(f"Features: {len(X_train.columns)} (Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)})")
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Class distribution - Train: {dict(y_train.value_counts())}")
        
        # Build robust models
        models = build_robust_model_pipelines(numeric_features, categorical_features, len(X_train.columns))
        param_grids = get_conservative_param_grids()
        dataset_results = []
        
        for model_name, pipeline in models.items():
            print(f"\nTraining and tuning {model_name} with overfitting prevention...")
            
            try:
                # Robust hyperparameter tuning
                if model_name in param_grids:
                    best_model, best_params, best_cv_score, val_score, overfitting_gap = tune_hyperparameters_robust(
                        pipeline, param_grids[model_name], X_train, y_train, X_val, y_val
                    )
                    print(f"   Best CV F1 Score: {best_cv_score:.4f}")
                    print(f"   Validation F1 Score: {val_score:.4f}")
                    print(f"   CV Overfitting Gap: {overfitting_gap:.4f}")
                    
                    if overfitting_gap > 0.1:
                        print(f"   ⚠️  WARNING: Potential overfitting detected (gap > 0.1)")
                    
                else:
                    # Just train the model without tuning
                    best_model = pipeline
                    best_model.fit(X_train, y_train)
                    best_cv_score = None
                    best_params = None
                    val_score = None
                
                # Comprehensive evaluation on test set
                metrics = evaluate_model_comprehensive(best_model, X_test, y_test, X_train, y_train, X_val, y_val)
                
                # Save results with overfitting information
                model_metrics = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc'],
                    'cv_f1_score': best_cv_score,
                    'val_f1_score': val_score,
                    'train_f1': metrics['train_f1'],
                    'val_f1': metrics['val_f1'],
                    'overfitting_gap': metrics['overfitting_gap'],
                    'is_overfitting': metrics['is_overfitting'],
                    'best_params': str(best_params) if best_params else None
                }
                dataset_results.append(model_metrics)
                full_results.append(model_metrics)
                
                # Save model with training plots for epoch-based models
                model_suffix = "OVERFIT" if metrics['is_overfitting'] else "ROBUST"
                model_path = f"saved_models/{dataset_name}_{model_name.replace(' ', '_')}_{model_suffix}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                
                # Generate and save training plots for epoch-based models
                if hasattr(best_model.named_steps['classifier'], 'plot_training_history'):
                    try:
                        plot_path = f"validation_plots/{dataset_name}_{model_name.replace(' ', '_')}_training_history.png"
                        fig = best_model.named_steps['classifier'].plot_training_history(save_path=plot_path)
                        if fig:
                            plt.close(fig)  # Close to save memory
                            print(f"   📊 Training history saved to {plot_path}")
                    except Exception as plot_error:
                        print(f"   ⚠️  Could not save training plot: {plot_error}")
                
                # Status reporting
                status_emoji = "⚠️" if metrics['is_overfitting'] else "✅"
                print(f"{status_emoji} Trained & saved {model_name}")
                print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Test F1: {metrics['f1']:.4f}")
                print(f"   Test AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "   Test AUC: N/A")
                print(f"   Train F1: {metrics['train_f1']:.4f}")
                print(f"   Val F1: {metrics['val_f1']:.4f}")
                print(f"   Overfitting Gap: {metrics['overfitting_gap']:.4f}")
                
                if metrics['is_overfitting']:
                    print(f"   🚨 OVERFITTING DETECTED - Model may not generalize well!")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Save dataset report with overfitting analysis
        if dataset_results:
            report_df = pd.DataFrame(dataset_results)
            report_df.to_csv(f"reports/{dataset_name}_robust_report.csv", index=False)
            
            # Find best NON-OVERFITTING model
            non_overfitting = report_df[~report_df['is_overfitting']]
            if len(non_overfitting) > 0:
                best_model_row = non_overfitting.loc[non_overfitting['f1'].idxmax()]
                print(f"\n🏆 Best ROBUST model for {dataset_name}: {best_model_row['model']}")
                print(f"   F1 Score: {best_model_row['f1']:.4f}")
                print(f"   Accuracy: {best_model_row['accuracy']:.4f}")
                print(f"   Overfitting Gap: {best_model_row['overfitting_gap']:.4f}")
            else:
                print(f"\n⚠️  All models show signs of overfitting for {dataset_name}")
                print("   Consider: more data, stronger regularization, or simpler models")
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")

# Generate final comprehensive report with overfitting analysis
if full_results:
    final_report = pd.DataFrame(full_results)
    final_report.to_csv("reports/full_robust_evaluation_report.csv", index=False)
    
    print(f"\n{'='*50}")
    print("FINAL ROBUST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    # Summary by dataset (focusing on non-overfitting models)
    for dataset in final_report['dataset'].unique():
        dataset_data = final_report[final_report['dataset'] == dataset]
        non_overfitting = dataset_data[~dataset_data['is_overfitting']]
        
        print(f"\n{dataset.upper()}:")
        if len(non_overfitting) > 0:
            best_model = non_overfitting.loc[non_overfitting['f1'].idxmax()]
            print(f"  Best ROBUST Model: {best_model['model']}")
            print(f"  F1 Score: {best_model['f1']:.4f}")
            print(f"  Accuracy: {best_model['accuracy']:.4f}")
            print(f"  Overfitting Gap: {best_model['overfitting_gap']:.4f}")
        else:
            print(f"  ⚠️  No robust models found - all show overfitting")
            overfitting_count = dataset_data['is_overfitting'].sum()
            print(f"  {overfitting_count}/{len(dataset_data)} models are overfitting")
    
    # Overall overfitting statistics
    total_models = len(final_report)
    overfitting_models = final_report['is_overfitting'].sum()
    overfitting_rate = (overfitting_models / total_models) * 100
    
    print(f"\n{'='*30}")
    print("OVERFITTING ANALYSIS:")
    print(f"{'='*30}")
    print(f"Total models trained: {total_models}")
    print(f"Models with overfitting: {overfitting_models}")
    print(f"Overfitting rate: {overfitting_rate:.1f}%")
    
    # Top 5 ROBUST models
    robust_models = final_report[~final_report['is_overfitting']]
    if len(robust_models) > 0:
        print(f"\n{'='*30}")
        print("TOP 5 ROBUST MODELS (by F1):")
        print(f"{'='*30}")
        top_robust = robust_models.nlargest(5, 'f1')
        for idx, row in top_robust.iterrows():
            print(f"{row['dataset']} - {row['model']}: F1={row['f1']:.4f}, Gap={row['overfitting_gap']:.4f}")
    else:
        print(f"\n⚠️  No robust models found across all datasets!")
        print("Consider:")
        print("- Collecting more training data")
        print("- Stronger regularization")
        print("- Simpler model architectures")
    
    print(f"\nFull robust evaluation report saved to reports/full_robust_evaluation_report.csv")
else:
    print("\nNo results generated due to errors")

print("\nRobust processing with overfitting prevention complete!")