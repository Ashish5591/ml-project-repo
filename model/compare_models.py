"""
compare_models.py
==================
Step 5: Model comparison.

Run this script AFTER all five model notebooks (logistic_regression.ipynb,
decision_tree.ipynb, knn.ipynb, naive_bayes.ipynb, random_forest.ipynb) have
been executed at least once. Each notebook appends its own evaluation
metrics (computed on the SAME test set, via `utils.evaluate_model` +
`utils.append_metrics`) to `results/metrics_store.csv`.

This script reads that shared store, builds the final comparison table
(Model, Accuracy, AUC, Precision, Recall, F1, MCC), sorts it by Accuracy
descending, prints it, and exports it as `results/comparison_table.csv`.

No metric values are invented here — everything is read directly from the
metrics that each notebook actually computed and persisted.

Usage
-----
    cd model
    python compare_models.py
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure this script can be run both as `python model/compare_models.py`
# (from project root) and as `python compare_models.py` (from inside model/).
sys.path.append(str(Path(__file__).resolve().parent))

from utils import METRICS_STORE_PATH, RESULTS_DIR

COMPARISON_OUTPUT_PATH = RESULTS_DIR / "comparison_table.csv"

EXPECTED_MODELS = {
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Random Forest",
}


def build_comparison_table() -> pd.DataFrame:
    """
    Load the shared metrics store and build the sorted comparison table.

    Returns
    -------
    pd.DataFrame
        Columns: Model, Accuracy, AUC, Precision, Recall, F1, MCC.
        Sorted by Accuracy descending.

    Raises
    ------
    FileNotFoundError
        If no model notebook has been run yet (metrics store missing).
    """
    if not METRICS_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Metrics store not found at '{METRICS_STORE_PATH}'. "
            "Run each model notebook first so it can append its metrics "
            "via utils.append_metrics()."
        )

    df = pd.read_csv(METRICS_STORE_PATH)

    missing = EXPECTED_MODELS - set(df["Model"])
    if missing:
        print(
            f"Warning: metrics are missing for: {sorted(missing)}. "
            "Run their notebooks before treating this comparison as final."
        )

    required_cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    df = df[required_cols]
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    """Build, display, and export the Step 5 model comparison table."""
    try:
        comparison_df = build_comparison_table()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("\nModel Comparison Table (sorted by Accuracy, descending):\n")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    print(f"\nComparison table exported to: {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
