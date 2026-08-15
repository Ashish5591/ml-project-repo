"""
utils.py
========
Shared utility functions for the Heart Disease UCI classification project.

These functions are imported by every model-training notebook
(logistic_regression.ipynb, decision_tree.ipynb, knn.ipynb, naive_bayes.ipynb,
random_forest.ipynb) so that ALL models are trained and evaluated on the
IDENTICAL preprocessed data, using the same random_state and the same
train/test split. 
"""

import io
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# All paths are relative to the project root (no hardcoded absolute paths).
# This file lives in <project_root>/model/utils.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_STORE_PATH = RESULTS_DIR / "metrics_store.csv"

for _directory in (DATA_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# The 13 clinical predictor columns used by the Heart Disease UCI dataset.
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"

# The combined 920-instance dataset is the concatenation of the four
# individual UCI "processed" source files (Cleveland 303 + Hungarian 294 +
# Switzerland 123 + VA Long Beach 200 = 920). NOTE: the `ucimlrepo` package's
# `fetch_ucirepo(id=45)` call only returns the 303-row Cleveland subset (that
# is what UCI's own metadata for id=45 lists as "# Instances: 303"), so it is
# NOT used here — we fetch the four raw source files directly instead.
_UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
_SOURCE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
# Column order used by every one of the four raw "processed.*.data" files
# (comma-separated, no header row, missing values encoded as "?").
_RAW_COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load_data(local_csv_name: str = "heart_disease_uci.csv") -> pd.DataFrame:
    """
    Load the combined Heart Disease UCI dataset (920 instances, 13 features,
    Cleveland + Hungarian + Switzerland + VA Long Beach sources).

    Loading strategy (in order of preference):
        1. Fetch all four raw "processed.*.data" source files directly from
           the UCI ML Repository archive and concatenate them (303 + 294 +
           123 + 200 = 920 rows), then cache the result locally as CSV.
        2. Fall back to a locally cached / manually placed CSV file at
           `data/<local_csv_name>`.

    The target column `num` (0 = no disease, 1-4 = increasing severity) is
    converted into a binary target: 0 = no disease, 1 = disease present.

    Parameters
    ----------
    local_csv_name : str
        Filename of the cached/local CSV inside the `data/` folder.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 13 feature columns + binary `target` column
        (920 rows when fetched fresh from all four sources).

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be fetched online and no local copy exists.
    KeyError
        If the target column cannot be located in the dataset.
    """
    local_path = DATA_DIR / local_csv_name
    df = None

    # --- Attempt 1: fetch and combine all four raw source files ---
    # A browser-like User-Agent is required: archive.ics.uci.edu returns
    # HTTP 403 Forbidden for requests without one (e.g. pandas' bare
    # pd.read_csv(url), which sends no User-Agent header at all).
    _request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        frames = []
        for source_name, filename in _SOURCE_FILES.items():
            url = f"{_UCI_BASE_URL}/{filename}"
            response = requests.get(url, headers=_request_headers, timeout=15)
            response.raise_for_status()
            frame = pd.read_csv(
                io.StringIO(response.text), header=None, names=_RAW_COLUMN_NAMES, na_values="?"
            )
            frame["source"] = source_name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(local_path, index=False)  # cache for future offline runs
        print(
            f"Dataset fetched from the UCI ML Repository (4 combined sources): "
            f"{len(df)} rows."
        )
    except Exception as exc:  
        print(f"Direct UCI fetch failed ({exc}). Falling back to local file...")

    # --- Attempt 2: local cached / manually downloaded CSV ---
    if df is None:
        if local_path.exists():
            df = pd.read_csv(local_path)
            print(f"Dataset loaded from local file: {local_path}")
        else:
            raise FileNotFoundError(
                "Could not fetch the dataset from the UCI ML Repository and "
                f"no local copy was found at '{local_path}'. Please "
                "download the four files (processed.cleveland.data, "
                "processed.hungarian.data, processed.switzerland.data, "
                "processed.va.data) from "
                "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/, "
                "concatenate them, save the result as "
                f"'{local_csv_name}', and place it in the 'data/' folder."
            )

    # --- Normalise column names (handles both freshly-fetched & cached CSVs) ---
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"thalach": "thalch"})

    # --- Build binary target from the multi-class `num` column ---
    if "num" in df.columns:
        df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])
    elif TARGET_COLUMN not in df.columns:
        raise KeyError(
            "Target column 'num' (or 'target') was not found in the dataset."
        )

    # --- Drop non-predictive identifier / source columns, if present ---
    drop_cols = [c for c in ("id", "dataset", "source") if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only known feature columns (+ target) that actually exist
    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + [TARGET_COLUMN]
    df = df[keep_cols]

    return df


# ---------------------------------------------------------------------------
# Step 1: Exploratory Data Analysis
# ---------------------------------------------------------------------------
def perform_eda(df: pd.DataFrame) -> None:
    """
    Print a structured Exploratory Data Analysis (EDA) summary.

    Displays: shape, columns, dtypes, missing values, duplicate count,
    target distribution, and descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The raw / loaded dataframe to summarise.
    """
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nShape of dataset (rows, columns): {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nNumber of duplicate records: {df.duplicated().sum()}")

    print("\nTarget distribution (0 = No Disease, 1 = Disease Present):")
    print(df[TARGET_COLUMN].value_counts())
    print(
        (df[TARGET_COLUMN].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        + " %"
    )

    print("\nDescriptive statistics:")
    with pd.option_context("display.max_columns", None):
        print(df.describe(include="all").T)


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Clean, encode, scale, and split the dataset into train/test sets.

    Steps performed:
        1. Impute missing numeric values with the median.
        2. Impute missing categorical values with the mode.
        3. Label-encode binary categorical columns.
        4. One-hot encode multi-category categorical columns.
        5. Train/test split (80:20, stratified on target, random_state=42).
        6. Standard-scale all features (fit on train, transform on both).
        7. Persist the held-out test set to `test_data.csv` for reproducibility.
        8. Persist the fitted scaler, imputers, and column metadata to
           `model/*.joblib` so app.py can apply IDENTICAL preprocessing to
           new raw CSV uploads at inference time.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `load_data()`.
    test_size : float, default=0.2
        Proportion of the dataset to allocate to the test split.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    df = df.copy()

    numeric_cols = [c for c in ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
                    if c in df.columns]
    categorical_cols = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
                         if c in df.columns]

    # --- Handle missing values ---
    num_imputer = None
    cat_imputer = None

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # --- Encoding ---
    binary_like_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    multi_cat_cols = [c for c in categorical_cols if df[c].nunique() > 2]

    for col in binary_like_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    if multi_cat_cols:
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # --- Train / test split ---
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # --- Feature scaling (fit on train only, to avoid data leakage) ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # --- Persist artifacts for reproducibility across notebooks / the app ---
    test_export = X_test_scaled.copy()
    test_export[TARGET_COLUMN] = y_test.values
    test_export.to_csv(TEST_DATA_PATH, index=False)

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(list(X_train_scaled.columns), MODEL_DIR / "feature_names.joblib")

    # Persist the fitted imputers (+ which raw columns each applies to) so
    # app.py can replicate the EXACT same missing-value handling on new,
    # raw CSV uploads at inference time. Without this, a raw upload with
    # missing values (e.g. missing 'ca'/'thal', common in this dataset)
    # would pass NaNs straight through to the model and fail at predict().
    if num_imputer is not None:
        joblib.dump(num_imputer, MODEL_DIR / "num_imputer.joblib")
    if cat_imputer is not None:
        joblib.dump(cat_imputer, MODEL_DIR / "cat_imputer.joblib")
    joblib.dump(numeric_cols, MODEL_DIR / "numeric_cols.joblib")
    joblib.dump(categorical_cols, MODEL_DIR / "categorical_cols.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X_train_scaled.columns)


# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series,
                    feature_names=None, save_plots: bool = True) -> dict:
    """
    Evaluate a trained classifier on the held-out test set.

    Computes Accuracy, ROC AUC, Precision, Recall, F1, and MCC. Also prints
    the confusion matrix and classification report, and (optionally) saves
    a confusion matrix plot, ROC curve plot, and feature-importance plot
    (for models that expose `feature_importances_` or `coef_`).

    Parameters
    ----------
    model : fitted sklearn estimator
    model_name : str
        Human-readable model name, used for print statements, filenames,
        and as the key in the shared metrics store.
    X_test, y_test : test features / labels
    feature_names : list[str], optional
        Required to plot feature importance.
    save_plots : bool, default=True
        Whether to save PNG plots to `results/plots/`.

    Returns
    -------
    dict
        Dictionary with keys: Model, Accuracy, AUC, Precision, Recall, F1, MCC.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # fallback; AUC will be degenerate but code won't crash

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    print(f"\n{'=' * 60}\nEvaluation Report: {model_name}\n{'=' * 60}")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if save_plots:
        safe_name = model_name.replace(" ", "_").lower()

        # Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_matrix_{safe_name}.png")
        plt.close(fig)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"roc_curve_{safe_name}.png")
        plt.close(fig)

        # Feature importance (only where the model exposes it)
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).ravel()

        if importances is not None and feature_names is not None:
            imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 5))
            imp_series.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_title(f"Feature Importance - {model_name}")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"feature_importance_{safe_name}.png")
            plt.close(fig)
        else:
            print(f"Note: '{model_name}' does not expose feature importances/coefficients.")

    return metrics


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_model(model, filename: str) -> None:
    """
    Persist a trained model to the `model/` directory using joblib.

    Parameters
    ----------
    model : fitted sklearn estimator
    filename : str
        Target filename, e.g. 'logistic_regression.joblib'.
    """
    try:
        path = MODEL_DIR / filename
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
    except Exception as exc:  
        raise IOError(f"Failed to save model '{filename}': {exc}") from exc


def append_metrics(metrics: dict) -> None:
    """
    Append (or replace, on re-run) a model's metrics as a row in the shared
    metrics store CSV (`results/metrics_store.csv`). This store is later
    consumed by `compare_models.py` to build the comparison table.

    Parameters
    ----------
    metrics : dict
        Output of `evaluate_model()`.
    """
    try:
        row_df = pd.DataFrame([metrics])
        if METRICS_STORE_PATH.exists():
            existing = pd.read_csv(METRICS_STORE_PATH)
            existing = existing[existing["Model"] != metrics["Model"]]
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df
        combined.to_csv(METRICS_STORE_PATH, index=False)
        print(f"Metrics for '{metrics['Model']}' appended to {METRICS_STORE_PATH}")
    except Exception as exc:  
        raise IOError(f"Failed to append metrics: {exc}") from exc
