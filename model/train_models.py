"""
train_models.py
================
Standalone, non-notebook training pipeline for all five classifiers.

This script is the recommended way to (re)generate every trained-model
artifact, the shared scaler, the held-out test set, and the model
comparison table in a single command-line run — no Jupyter required.
This matters for deployment: Streamlit Community Cloud (and most
deployment platforms) cannot execute .ipynb notebooks as part of a build
step, so this script is what actually produces the artifacts app.py loads.

It performs, end to end and in order, exactly what Steps 1-5 describe:
    Step 1: Load dataset + EDA
    Step 2: Preprocessing (imputation, encoding, scaling, 80:20 stratified split)
    Step 3: Train Logistic Regression, Decision Tree, KNN, Naive Bayes,
            Random Forest
    Step 4: Evaluate every model on the SAME held-out test set
    Step 5: Build and export the sorted model comparison table

It reuses the exact same functions from model/utils.py and
model/compare_models.py that the five notebooks use, so results are
identical whether you run the notebooks individually or this single script.

Usage
-----
    python train_models.py
"""

import sys
from pathlib import Path

# Make model/utils.py and model/compare_models.py importable from the
# project root (this file lives at <project_root>/train_models.py)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "model"))

from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  

from compare_models import COMPARISON_OUTPUT_PATH, build_comparison_table  
from utils import (  
    RANDOM_STATE,
    append_metrics,
    evaluate_model,
    load_data,
    perform_eda,
    preprocess_data,
    save_model,
)

# ---------------------------------------------------------------------------
# Registry of all five models: constructor + artifact filename.
# Using lambdas defers instantiation until train_all_models() actually runs.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": {
        "estimator": lambda: LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "filename": "logistic_regression.joblib",
    },
    "Decision Tree": {
        "estimator": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "filename": "decision_tree.joblib",
    },
    "K-Nearest Neighbors": {
        "estimator": lambda: KNeighborsClassifier(n_neighbors=5),  # no random_state param
        "filename": "knn.joblib",
    },
    "Naive Bayes": {
        "estimator": lambda: GaussianNB(),  # no random_state param
        "filename": "naive_bayes.joblib",
    },
    "Random Forest": {
        "estimator": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "filename": "random_forest.joblib",
    },
}


def train_all_models() -> None:
    """Run Steps 1-5 end to end for every model in MODEL_REGISTRY."""
    print("Step 1: Loading dataset and running EDA...")
    df = load_data()
    perform_eda(df)

    print("\nStep 2: Preprocessing dataset (identical split shared by all models)...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")

    for model_name, spec in MODEL_REGISTRY.items():
        print(f"\n{'-' * 60}\nStep 3: Training {model_name}...\n{'-' * 60}")
        model = spec["estimator"]()
        model.fit(X_train, y_train)

        print(f"Step 4: Evaluating {model_name} on the held-out test set...")
        metrics = evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )

        save_model(model, spec["filename"])
        append_metrics(metrics)

    print(f"\n{'=' * 60}\nStep 5: Building model comparison table...\n{'=' * 60}")
    comparison_df = build_comparison_table()
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    print(f"\nComparison table exported to: {COMPARISON_OUTPUT_PATH}")
    print("\nAll models trained, evaluated, and saved successfully.")


if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as exc:  
        print(f"Training pipeline failed: {exc}")
        raise
