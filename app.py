"""
app.py
======
Streamlit dashboard for the Heart Disease UCI Classification project.

Implements Step 6 and Step 9 of the project spec:

Sidebar:
    * Model selection dropdown
    * Upload CSV button
    * Predict button

Main Page:
    * Uploaded dataset preview
    * Dataset information (shape, dtypes, missing values, duplicates)
    * Selected model
    * Predictions
    * Accuracy / Precision / Recall / F1 / AUC / MCC as Streamlit metric cards
    * Confusion Matrix (matplotlib)
    * Classification Report

Additionally, a permanent "Model Performance on Held-Out Test Set" section
evaluates the selected model live against the project's real test_data.csv
(produced by train_models.py or the notebooks), so genuine test-set results
are always visible in the app, independent of any file upload (Step 9).

No metric value anywhere in this file is hardcoded or invented: every
number shown is computed at runtime by scikit-learn from real predictions
against real data.

Run with:
    streamlit run app.py
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path configuration (relative to this file — no hardcoded absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.csv"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

TARGET_COLUMN = "target"
RAW_FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]
BINARY_RAW_COLUMNS = ["sex", "fbs", "exang"]
MULTI_CAT_RAW_COLUMNS = ["cp", "restecg", "slope", "thal"]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load a trained model artifact (joblib) from the model/ directory."""
    path = MODEL_FILES.get(model_name)
    if path is not None and path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            st.error(f"Failed to load model '{model_name}': {exc}")
            return None
    return None


@st.cache_resource(show_spinner=False)
def load_scaler_and_features():
    """Load the shared fitted StandardScaler and training-time feature order."""
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    feature_names = joblib.load(FEATURE_NAMES_PATH) if FEATURE_NAMES_PATH.exists() else None
    return scaler, feature_names


@st.cache_data(show_spinner=False)
def load_test_data() -> Optional[pd.DataFrame]:
    """Load the shared held-out test set (already scaled/encoded + target)."""
    if TEST_DATA_PATH.exists():
        try:
            df = pd.read_csv(TEST_DATA_PATH)
        except Exception:
            return None
        if not df.empty and TARGET_COLUMN in df.columns:
            return df
    return None


@st.cache_data(show_spinner=False)
def load_comparison_table() -> Optional[pd.DataFrame]:
    """Load the Step 5 model-comparison table, if it has been generated."""
    if COMPARISON_TABLE_PATH.exists():
        return pd.read_csv(COMPARISON_TABLE_PATH)
    return None


# ---------------------------------------------------------------------------
# Shared computation / rendering helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute Accuracy, Precision, Recall, F1, AUC, MCC from real predictions."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        # Happens if y_true contains only one class (e.g. a tiny/filtered upload)
        metrics["AUC"] = float("nan")
    return metrics


def render_metric_cards(metrics: dict) -> None:
    """Render Accuracy/Precision/Recall/F1/AUC/MCC as Streamlit metric cards."""
    order = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
    cols = st.columns(len(order))
    for col, name in zip(cols, order):
        value = metrics.get(name)
        col.metric(name, f"{value:.4f}" if pd.notna(value) else "N/A")


def render_confusion_matrix(y_true, y_pred, title: str) -> None:
    """Plot a confusion matrix with matplotlib and render it in Streamlit."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_classification_report(y_true, y_pred) -> None:
    """Render the sklearn classification report as a formatted DataFrame."""
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)


def encode_and_scale_raw(df: pd.DataFrame, feature_names: list, scaler) -> pd.DataFrame:
    """
    Apply the exact same encoding/scaling used during training to a batch
    of RAW (unencoded) uploaded records, and align columns to the
    training-time feature order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature columns only (no target column).
    feature_names : list[str]
        Column order the model/scaler were trained on.
    scaler : fitted StandardScaler

    Returns
    -------
    pd.DataFrame
        Scaled, correctly-ordered DataFrame ready for `.predict()`.
    """
    df = df.copy()

    for col in BINARY_RAW_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    present_multi_cat = [c for c in MULTI_CAT_RAW_COLUMNS if c in df.columns]
    if present_multi_cat:
        df = pd.get_dummies(df, columns=present_multi_cat, drop_first=True)

    df = df.reindex(columns=feature_names, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(df), columns=feature_names, index=df.index)
    return scaled


def extract_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return (features_df, target_series_or_None) by detecting a target/num column."""
    for candidate in (TARGET_COLUMN, "num"):
        if candidate in df.columns:
            y = df[candidate]
            if candidate == "num":
                y = (y > 0).astype(int)
            X = df.drop(columns=[candidate])
            return X, y
    return df, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render sidebar controls: model dropdown, CSV upload, Predict button."""
    st.sidebar.header("Configuration")

    model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "Expected columns: age, sex, cp, trestbps, chol, fbs, restecg, "
            "thalch, exang, oldpeak, slope, ca, thal, and optionally a "
            "'target' (0/1) or 'num' column for metric evaluation."
        ),
    )

    predict_clicked = st.sidebar.button("Predict", type="primary")

    return model_name, uploaded_file, predict_clicked


# ---------------------------------------------------------------------------
# Main-page sections
# ---------------------------------------------------------------------------
def render_test_data_section(model_name: str) -> None:
    """
    Step 9: Always-visible section showing the selected model's real
    performance on the shared held-out test set (test_data.csv), computed
    live at runtime.
    """
    st.header("Model Performance on Held-Out Test Set")

    comparison_df = load_comparison_table()
    if comparison_df is not None:
        st.subheader("All 5 Models — Comparison Table")
        st.dataframe(
            comparison_df.style.format(
                {c: "{:.4f}" for c in comparison_df.columns if c != "Model"}
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "No comparison table found yet. Run 'python train_models.py' "
            "(or all five notebooks + model/compare_models.py) to generate "
            "results/comparison_table.csv."
        )

    test_df = load_test_data()
    model = load_model(model_name)

    if test_df is None:
        st.warning(
            "No held-out test set found yet at 'test_data.csv'. Run "
            "'python train_models.py' (or the notebooks) first."
        )
        return
    if model is None:
        st.warning(f"No saved model found for '{model_name}'. Run 'python train_models.py' first.")
        return

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    except Exception as exc:
        st.error(f"Could not evaluate '{model_name}' on the test set: {exc}")
        return

    st.subheader(f"Live Evaluation — {model_name} on test_data.csv ({len(test_df)} rows)")
    metrics = compute_metrics(y_test, y_pred, y_proba)
    render_metric_cards(metrics)

    col_cm, col_report = st.columns([1, 1.3])
    with col_cm:
        st.markdown("**Confusion Matrix**")
        render_confusion_matrix(y_test, y_pred, f"{model_name} — Test Data")
    with col_report:
        st.markdown("**Classification Report**")
        render_classification_report(y_test, y_pred)


def render_upload_predict_section(model_name: str, uploaded_file, predict_clicked: bool) -> None:
    """Step 6: Upload CSV -> preview/info -> predict -> metrics/confusion matrix/report."""
    st.header("Upload & Predict")

    if uploaded_file is None:
        st.info("Upload a CSV from the sidebar, then click 'Predict'.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        return

    # --- Uploaded dataset preview ---
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # --- Dataset information ---
    st.subheader("Dataset Information")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"Shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
        st.write("Data types:")
        st.dataframe(raw_df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with info_col2:
        st.write("Missing values per column:")
        st.dataframe(raw_df.isnull().sum().rename("missing_count"), use_container_width=True)
        st.write(f"Duplicate rows: {raw_df.duplicated().sum()}")

    # --- Selected model ---
    st.subheader(f"Selected Model: {model_name}")

    if not predict_clicked:
        st.info("Click 'Predict' in the sidebar to run inference on this file.")
        return

    model = load_model(model_name)
    scaler, feature_names = load_scaler_and_features()

    if model is None or scaler is None or feature_names is None:
        st.warning(
            f"Artifacts for '{model_name}' are not available yet. Run "
            "'python train_models.py' (or the corresponding notebook) first."
        )
        return

    features_df, y_true = extract_target(raw_df)

    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in features_df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
        return

    try:
        X_scaled = encode_and_scale_raw(features_df[RAW_FEATURE_COLUMNS], feature_names, scaler)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    # --- Predictions ---
    st.subheader("Predictions")
    results_df = raw_df.copy()
    results_df["Predicted"] = pd.Series(y_pred, index=results_df.index).map(
        {0: "No Disease", 1: "Disease Present"}
    )
    results_df["Probability (Disease)"] = y_proba
    st.dataframe(results_df, use_container_width=True)

    if y_true is None:
        st.info(
            "No 'target' or 'num' column found in the upload, so Accuracy / "
            "Precision / Recall / F1 / AUC / MCC and the confusion matrix "
            "cannot be computed. Predictions above are still valid."
        )
        return

    # --- Metrics, confusion matrix, classification report ---
    st.subheader("Evaluation Metrics")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    render_metric_cards(metrics)

    col_cm, col_report = st.columns([1, 1.3])
    with col_cm:
        st.markdown("**Confusion Matrix**")
        render_confusion_matrix(y_true, y_pred, f"{model_name} — Uploaded Data")
    with col_report:
        st.markdown("**Classification Report**")
        render_classification_report(y_true, y_pred)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(page_title="Heart Disease Risk Classifier", layout="wide")
    st.title("Heart Disease UCI — Classification Dashboard")
    st.caption(
        "Binary classification (Disease Present vs. No Disease) on the "
        "Heart Disease UCI dataset (920 instances, 13 features)."
    )

    model_name, uploaded_file, predict_clicked = render_sidebar()

    render_test_data_section(model_name)
    st.divider()
    render_upload_predict_section(model_name, uploaded_file, predict_clicked)


if __name__ == "__main__":
    main()