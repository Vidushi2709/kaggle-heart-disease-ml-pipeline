# Heart Disease Risk Prediction

**Kaggle Playground Series 2026**

---

## Overview

This repository contains my solution to the **2026 Kaggle Playground Series: Heart Disease Prediction** challenge. The objective is to predict the **likelihood of heart disease** from structured clinical data by producing calibrated probability estimates rather than hard diagnostic decisions.

The solution follows an **end-to-end machine learning workflow**, including exploratory data analysis (EDA), feature preparation, model training, Optuna-based hyperparameter optimization, cross-validated ensembling, and Kaggle-ready submission generation.

---

## Problem Objective

* **Input:** Patient-level clinical features
* **Output:** Probability of heart disease
* **Evaluation:** Kaggle competition metric
* **Approach:** Supervised learning with probabilistic predictions

---

## Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── submissions/
│   ├── submission.csv
│   ├── submission_optuna_*.csv
│   └── submissions.csv
│
├── heart-disease-detection.ipynb
└── README.md
```

---

## Notebook Workflow

### 1. Exploratory Data Analysis (EDA)

* Dataset inspection and sanity checks
* Feature distributions and correlations
* Target behavior analysis

### 2. Data Preparation

* Feature–target separation
* Handling numerical and categorical features
* Stratified train–validation splits

### 3. Modeling

The following gradient-boosted models are used:

* XGBoost
* LightGBM
* CatBoost

Each model produces **probabilistic outputs**, aligned with the competition objective.

### 4. Hyperparameter Optimization

* Optuna-based systematic tuning
* Optimization performed using stratified cross-validation

### 5. Ensembling and Validation

* Out-of-fold (OOF) predictions
* Ensemble averaging for improved robustness
* Fold-wise validation tracking

### 6. Inference and Submission

* Final training on the full dataset
* Probability predictions on the test set
* Kaggle-compatible submission files

---

## Key Techniques Used

* Gradient Boosted Decision Trees (GBDT)
* Probabilistic risk prediction
* Stratified K-Fold Cross-Validation
* Optuna-based hyperparameter tuning
* Model ensembling

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/Vidushi2709/kaggle-heart-disease-ml-pipeline.git
cd kaggle-heart-disease-ml-pipeline
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook heart-disease-detection.ipynb
```

---

## Results

The ensemble approach provides **stable and competitive cross-validated performance**, making it a strong baseline for predicting heart disease risk in the Playground Series setting.

Exact leaderboard scores may vary depending on random seeds and optimization trials.

---

## Future Improvements

* SHAP-based model explainability
* Risk calibration and threshold analysis
* Uncertainty-aware prediction / abstention
* Safety-focused validation layers (e.g., EBM filtering)

---

## Competition

* Kaggle Playground Series – 2026
* Task: Heart Disease Risk Prediction
