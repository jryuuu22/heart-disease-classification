# Predicting Heart Disease Using Machine Learning

This project builds and evaluates machine learning models to predict whether a patient has heart disease based on clinical attributes. The notebook walks through a full applied ML workflow: data exploration, preprocessing, model comparison, hyperparameter tuning, evaluation beyond accuracy, and feature importance.

---

## Problem Definition

**Goal:**  
Given clinical parameters about a patient, can we predict whether or not they have heart disease?

---

## Data

**Primary source:** Cleveland Heart Disease dataset (UCI Machine Learning Repository)  
https://archive.ics.uci.edu/ml/datasets/heart+Disease

**Alternative source:** Kaggle dataset (convenient CSV format)  
https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset

**Dataset shape:** `303 rows × 15 columns` (including index-like column)

Target label is stored as `target` (yes/no in the raw CSV).

---

## Evaluation Metric

Success criteria for the proof-of-concept was:

- **Target:** 95% accuracy

In practice, the notebook evaluates models with **accuracy**, but also includes:
- ROC curve + AUC
- confusion matrix
- precision / recall / F1
- cross-validated metrics (5-fold CV)

These additional metrics matter because medical classification often has uneven costs (false negatives vs false positives).

---

## Features (Data Dictionary)

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female in many versions; this notebook converts male→0, female→1) |
| `cp` | Chest pain type: 0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic |
| `trestbps` | Resting blood pressure (mm Hg); \>130–140 can be concerning |
| `chol` | Serum cholesterol (mg/dL); \>200 can be concerning |
| `fbs` | Fasting blood sugar \>120 mg/dL (1=true, 0=false); \>126 mg/dL suggests diabetes |
| `restecg` | Resting ECG results: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1=yes, 0=no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (0 upsloping, 1 flat, 2 downsloping) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia / thallium stress test result (values vary by dataset version) |
| `target` | Heart disease presence (raw dataset stores yes/no; notebook encodes to binary) |

---

## Tools & Libraries

- Data: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Modeling: `scikit-learn`
  - `LogisticRegression`
  - `KNeighborsClassifier`
  - `RandomForestClassifier`
- Evaluation:
  - confusion matrix, classification report
  - ROC curve + AUC
  - cross-validation (`cross_val_score`)
- Tuning:
  - `RandomizedSearchCV`
  - `GridSearchCV`

---

## Workflow Overview

### 1) Data Loading & Cleaning
- Loads CSV (`303 × 15`)
- Checks missing values (`trestbps`, `chol`, `thalach` contain some NaNs)
- Converts categorical text:
  - `sex`: male→0, female→1
  - `target`: yes/no → binary encoding (note: encoding direction matters—see “Notes” below)
- Drops missing rows (`df.dropna(axis=0)`)
- Drops index-like column `Unnamed: 0`

### 2) Exploratory Data Analysis (EDA)
Highlights include:
- Class balance check (`yes` vs `no`)
- Heart disease frequency by sex (cross-tab + bar chart)
- Scatter: age vs max heart rate colored by disease
- Heart disease frequency by chest pain type
- Correlation matrix heatmap

### 3) Baseline Model Comparison
Models trained and compared on a train/test split:
- Logistic Regression
- KNN
- Random Forest

Initial test accuracies (from your run):
- Logistic Regression: **~0.797**
- KNN: **~0.610**
- Random Forest: **~0.831**

### 4) Hyperparameter Tuning
- KNN tuned by varying `n_neighbors` (1–20)
- Logistic Regression tuned with `RandomizedSearchCV` and `GridSearchCV`
- Random Forest tuned with `RandomizedSearchCV`

Best RandomForest test accuracy after tuning reached **~0.864** in your notebook.

### 5) Evaluation Beyond Accuracy
For the tuned classifier, the notebook generates:
- ROC curve + AUC
- confusion matrix (Seaborn heatmap)
- classification report
- cross-validated accuracy / precision / recall / F1

Cross-validated metrics (5-fold) for tuned Logistic Regression (means from your run):
- Accuracy: **~0.850**
- Precision: **~0.891**
- Recall: **~0.758**
- F1: **~0.817**

### 6) Feature Importance / Interpretability
- Logistic Regression coefficients are used as a proxy for feature influence.
- A bar chart visualizes coefficient magnitudes and direction.

---

## Results Summary

- Best tuned accuracy observed: **~0.864** (Random Forest via RandomizedSearchCV)
- The original 95% accuracy goal was **not reached**, but the notebook demonstrates:
  - solid baseline performance
  - tuning improvements
  - proper evaluation using multiple metrics
  - interpretable coefficient-based insights

---
