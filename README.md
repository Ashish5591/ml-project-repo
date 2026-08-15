# Heart Disease UCI — Binary Classification Project

## a. Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, and
early, low-cost risk screening can meaningfully improve patient outcomes.
This project builds and compares five supervised machine learning
classifiers that predict whether a patient has heart disease (binary:
disease present vs. no disease) from 13 routinely-collected clinical
attributes (e.g., age, sex, chest pain type, resting blood pressure,
cholesterol, ECG results, exercise-induced angina, and more). The goal is
to identify which model provides the most reliable classification
performance on held-out data, and to expose that model through an
interactive Streamlit application for exploration and inference.

## b. Dataset Description

* **Source:** Heart Disease UCI dataset, combining four data sources —
  Cleveland (303), Hungary (294), Switzerland (123), and the VA Long Beach
  (200) — fetched directly from the UCI ML Repository archive.
* **Instances:** 920
* **Features:** 13 clinical attributes — `age`, `sex`, `cp` (chest pain
  type), `trestbps` (resting blood pressure), `chol` (serum cholesterol),
  `fbs` (fasting blood sugar > 120 mg/dl), `restecg` (resting ECG
  results), `thalch` (max heart rate achieved), `exang` (exercise-induced
  angina), `oldpeak` (ST depression), `slope` (slope of peak exercise ST
  segment), `ca` (number of major vessels colored by fluoroscopy), `thal`
  (thalassemia).
* **Target:** Binary — derived from the original multi-class `num` column
  (0 = no disease, 1-4 = increasing severity), collapsed to `0` = No
  Disease and `1` = Disease Present.
* **Task type:** Binary classification.
* **Split:** 80:20 stratified train/test split, `random_state=42`, shared
  identically across all five models.

## c. Github Repository Link

*https://github.com/Ashish5591/ml-project-repo*

## d. Machine Learning Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

All five models were trained on an identical 80:20 stratified train/test
split (`random_state=42`) and evaluated on the identical held-out test
set, so the comparison below is an apples-to-apples comparison. Values
are taken directly from `results/comparison_table.csv`.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Naive Bayes | 0.8478 | 0.8948 | 0.8364 | 0.9020 | 0.8679 | 0.6918 |
| Logistic Regression | 0.8424 | 0.9036 | 0.8411 | 0.8824 | 0.8612 | 0.6801 |
| K-Nearest Neighbors | 0.8370 | 0.8907 | 0.8333 | 0.8824 | 0.8571 | 0.6691 |
| Random Forest (Ensemble) | 0.8207 | 0.9162 | 0.8224 | 0.8627 | 0.8421 | 0.6358 |
| Decision Tree | 0.7663 | 0.7617 | 0.7810 | 0.8039 | 0.7923 | 0.5256 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong, well-balanced performer — second-highest Accuracy (0.8424) and F1 (0.8612), and the second-best AUC (0.9036). As a linear model it benefits from the scaled features and shows no signs of overfitting, making it a stable, interpretable baseline. |
| Decision Tree | Clearly the weakest model on every metric (Accuracy 0.7663, AUC 0.7617, MCC 0.5256). A single unpruned tree tends to overfit the training data and generalizes poorly to the test set; restricting `max_depth`/`min_samples_leaf` or switching to an ensemble (as Random Forest does) would likely close this gap. |
| K-Nearest Neighbors | Solid, closely trailing Logistic Regression (Accuracy 0.8370, F1 0.8571). Performance is sensitive to feature scaling (already applied here) and the choice of `k`; tuning `n_neighbors` via cross-validation could improve results further. |
| Naive Bayes | The top performer on Accuracy (0.8478), Recall (0.9020), F1 (0.8679), and MCC (0.6918) — despite its simplifying assumption of feature independence, which the 13 clinical features apparently do not violate too severely. Its high Recall is particularly valuable in a medical screening context, where missing an actual disease case (a false negative) is costlier than a false alarm. |
| Random Forest (Ensemble) | Achieves the best AUC (0.9162), indicating the strongest overall ranking/separation between classes across all thresholds, but its Accuracy, F1, and MCC trail Naive Bayes, Logistic Regression, and KNN at the default 0.5 threshold. This gap suggests the model is well-calibrated for ranking but may benefit from threshold tuning or hyperparameter search (e.g., `max_depth`, `min_samples_leaf`, `n_estimators`) to convert its ranking strength into better hard-classification metrics. |
| Overall Winner for the dataset | **Naive Bayes** — it leads on four of the six metrics (Accuracy, Precision-adjacent Recall, F1, and MCC), including the highest Recall, which matters most for disease screening. Random Forest is the strongest runner-up by AUC and is worth revisiting with threshold/hyperparameter tuning, but based on the metrics as computed, Naive Bayes is the best all-around model on this dataset. |
