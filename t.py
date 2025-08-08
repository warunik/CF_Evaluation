import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from xgboost import XGBClassifier

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
classes = iris.target_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Save train/test datasets
pd.DataFrame(X_train, columns=feature_names).assign(target=y_train).to_csv("iris_train.csv", index=False)
pd.DataFrame(X_test, columns=feature_names).assign(target=y_test).to_csv("iris_test.csv", index=False)

# Define models with given parameters
models = {
    "Logistic Regression": LogisticRegression(C=1, penalty="l1", solver="liblinear", random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=2, random_state=42),
    "MLP": MLPClassifier(activation="relu", alpha=0.001, hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# DataFrame for final results
results = []

# Loop through models
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    # AUC scores & binarization
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        pr_auc = average_precision_score(y_test_bin, y_score, average='weighted')

        # ROC curve plot
        plt.figure(figsize=(6, 5))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc_score(y_test_bin[:, i], y_score[:, i]):.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"roc_curve_{name.replace(' ', '_').lower()}.png")
        plt.close()

        # Precision-Recall curve plot
        plt.figure(figsize=(6, 5))
        for i in range(len(classes)):
            precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall_vals, precision_vals, label=f"Class {classes[i]} (AP = {average_precision_score(y_test_bin[:, i], y_score[:, i]):.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(f"precision_recall_curve_{name.replace(' ', '_').lower()}.png")
        plt.close()

    else:
        roc_auc = None
        pr_auc = None

    # Save model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_iris.pkl")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy',
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='green', label="Cross-validation score")
    plt.title(f"Learning Curve - {name} (Iris)")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"learning_curve_{name.replace(' ', '_').lower()}.png")
    plt.close()

    # Append results
    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 3),
        "Test Accuracy": round(test_acc, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3) if roc_auc is not None else None,
        "PR-AUC": round(pr_auc, 3) if pr_auc is not None else None
    })

# Save final CSV report
results_df = pd.DataFrame(results)
results_df.to_csv("iris_models_report.csv", index=False)
print("\nFinal report saved as 'iris_models_report.csv'")
