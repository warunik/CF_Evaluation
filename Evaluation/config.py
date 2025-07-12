# config.py
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
        },
        "model_paths": {
            "decision_tree": "Evaluation/saved_models/heart_Decision_Tree_tuned.pkl",
            "logistic_regression": "Evaluation/saved_models/heart_Logistic_Regression_tuned.pkl",
            "mlp": "Evaluation/saved_models/heart_MLP_tuned.pkl",
            "random_forest": "Evaluation/saved_models/heart_Random_Forest_tuned.pkl",
            "xgboost": "Evaluation/saved_models/heart_XGBoost_tuned.pkl"
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
        },
        "model_paths": {
            "decision_tree": "Evaluation/saved_models/diabetes_Decision_Tree_tuned.pkl",
            "logistic_regression": "Evaluation/saved_models/diabetes_Logistic_Regression_tuned.pkl",
            "mlp": "Evaluation/saved_models/diabetes_MLP_tuned.pkl",
            "random_forest": "Evaluation/saved_models/diabetes_Random_Forest_tuned.pkl",
            "xgboost": "Evaluation/saved_models/diabetes_XGBoost_tuned.pkl"
        }
    },
    "adult": {
        "name": "Income Prediction",
        "path": "Evaluation/data/adult.csv",
        "target_column": "class",
        "drop_columns": ["fnlwgt", "education-num", "native-country"],
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
        },
        "model_paths": {
            "decision_tree": "Evaluation/saved_models/adult_Decision_Tree_tuned.pkl",
            "logistic_regression": "Evaluation/saved_models/adult_Logistic_Regression_tuned.pkl",
            "mlp": "Evaluation/saved_models/adult_MLP_tuned.pkl",
            "random_forest": "Evaluation/saved_models/adult_Random_Forest_tuned.pkl",
            "xgboost": "Evaluation/saved_models/adult_XGBoost_tuned.pkl"
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
        },
        "model_paths": {
            "decision_tree": "Evaluation/saved_models/bank_Decision_Tree_tuned.pkl",
            "logistic_regression": "Evaluation/saved_models/bank_Logistic_Regression_tuned.pkl",
            "mlp": "Evaluation/saved_models/bank_MLP_tuned.pkl",
            "random_forest": "Evaluation/saved_models/bank_Random_Forest_tuned.pkl",
            "xgboost": "Evaluation/saved_models/bank_XGBoost_tuned.pkl"
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
        },
        "model_paths": {
            "decision_tree": "Evaluation/saved_models/german_Decision_Tree_tuned.pkl",
            "logistic_regression": "Evaluation/saved_models/german_Logistic_Regression_tuned.pkl",
            "mlp": "Evaluation/saved_models/german_MLP_tuned.pkl",
            "random_forest": "Evaluation/saved_models/german_Random_Forest_tuned.pkl",
            "xgboost": "Evaluation/saved_models/german_XGBoost_tuned.pkl"
        }
    }
}

ML_MODELS = {
    "mlp": "Multi-layer Perceptron",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
}