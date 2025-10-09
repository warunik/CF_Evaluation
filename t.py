import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def run_adult_models():
    # Load the Adult dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    data = pd.read_csv(url, names=columns, sep=',\s*', engine='python', na_values='?')
    data.dropna(inplace=True)

    # Separate features and target
    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Split categorical and numerical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models with specified hyperparameters
    models = {
        'Logistic Regression': LogisticRegression(
            C=1, penalty="l1", solver="liblinear", max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion="gini", max_depth=7, min_samples_split=5, random_state=42
        ),
        'MLP': MLPClassifier(
            activation="relu", alpha=0.0001, hidden_layer_sizes=(50,), 
            max_iter=1000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            max_depth=10, min_samples_split=5, n_estimators=200, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.1, max_depth=4, n_estimators=200, random_state=42
        ),
        'XGBoost': XGBClassifier(
            learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    }

    results = []

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Fit the model
        pipe.fit(X_train, y_train)

        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_prob)
        pr_auc = average_precision_score(y_test, y_test_prob)

        # Store results
        results.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 3),
            "Test Accuracy": round(test_acc, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-score": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC": round(pr_auc, 3)
        })

    # Create and print results DataFrame in CSV format
    results_df = pd.DataFrame(results)
    print("\n=== ADULT DATASET MODEL RESULTS ===")
    print(results_df.to_csv(index=False))
    
    return results_df

# Run the models
if __name__ == "__main__":
    run_adult_models()