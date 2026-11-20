# Stroke Prediction using Machine Learning

## Project Overview

This project predicts whether a patient is likely to experience a stroke using
demographic and medical information. It is a binary classification problem where:

- `0` means no stroke
- `1` means stroke

The main objective is to build a baseline machine-learning pipeline and
understand how class imbalance affects model evaluation.

## Dataset

The dataset contains patient-level records with the following features:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`
- `stroke` — target variable

The `id` column was removed because it is only a record identifier and does not
help the model make predictions.

## Preprocessing

The following preprocessing steps were performed:

- Removed the `id` column
- Imputed missing values in the `bmi` column using the mean
- One-hot encoded categorical columns:
  - `gender`
  - `ever_married`
  - `work_type`
  - `Residence_type`
  - `smoking_status`
- Scaled continuous numerical columns:
  - `age`
  - `avg_glucose_level`
  - `bmi`
- Kept `hypertension` and `heart_disease` as binary numerical features
- Used a stratified train-test split to preserve the stroke/non-stroke class ratio

A `ColumnTransformer` and `Pipeline` were used so that preprocessing steps are
fitted only on the training data and applied consistently to the test data.

## Model

A Logistic Regression model was trained with:

```python
LogisticRegression(class_weight="balanced")
```

The `class_weight="balanced"` parameter was used because the dataset is highly
imbalanced: most patients do not have a stroke, while only a small percentage do.

## Results

The model was evaluated on a stratified test set.

| Metric | No Stroke (0) | Stroke (1) |
|---|---:|---:|
| Precision | 0.99 | 0.14 |
| Recall | 0.74 | 0.80 |
| F1-score | 0.85 | 0.24 |
| Support | 972 | 50 |

- Accuracy: 75%
- ROC-AUC: 0.844

### Interpretation

The model correctly identifies 80% of actual stroke cases, which is important
because missing a stroke case can be costly. However, precision for the stroke
class is low, meaning the model produces many false positives.

The high overall accuracy in earlier experiments was misleading because the
dataset is dominated by non-stroke cases. A model can achieve high accuracy by
predicting “no stroke” for most patients while missing many actual stroke cases.

## Key Learning

For imbalanced medical datasets, accuracy alone is not a reliable metric.
Precision, recall, F1-score, the confusion matrix, and ROC-AUC provide a better
understanding of model performance.

## Future Improvements

Possible improvements include:

- Tuning the classification threshold to balance precision and recall
- Using stratified K-fold cross-validation
- Comparing Logistic Regression with Random Forest and XGBoost
- Handling class imbalance using resampling techniques
- Performing feature importance analysis
- Evaluating the model using a confusion matrix and precision-recall curve

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Main libraries used:

- pandas
- numpy
- scikit-learn

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload or provide the path to `healthcare-dataset-stroke-data.csv`.
3. Run all cells in order.
4. Review the model evaluation metrics and confusion matrix.

## Dataset Source

The dataset used in this project is the Healthcare Stroke Dataset.
