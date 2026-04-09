# Project 3 — Machine Learning Foundations

Heart Disease Prediction using the UCI Heart Disease dataset.

## Overview

This project implements a complete machine learning workflow for predicting heart disease from clinical features. A Random Forest Classifier (tuned via GridSearchCV) achieved **85.0% accuracy** and a **0.951 ROC-AUC** on the held-out test set.

## Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Records:** 303 patients (297 after removing missing values)
- **Features:** 13 clinical attributes (age, chest pain type, cholesterol, max heart rate, etc.)
- **Target:** Binary — presence (1) or absence (0) of heart disease

## Project Structure

```
├── modeling.ipynb          # Jupyter Notebook — full ML workflow
├── modeling.html           # HTML export of the executed notebook
├── module_summary.md       # Machine Learning Analysis Report
├── heart.csv               # Dataset (CSV)
├── requirements.txt        # Python dependencies (pip freeze)
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open and run `modeling.ipynb` top-to-bottom in Jupyter Notebook or VS Code.

## Key Results

| Metric    | Logistic Regression | Random Forest |
|-----------|---------------------|---------------|
| Accuracy  | 0.833               | 0.850         |
| Precision | 0.846               | 0.880         |
| Recall    | 0.786               | 0.786         |
| F1 Score  | 0.815               | 0.830         |
| ROC-AUC   | 0.950               | 0.951         |

## Tools Used

- Python 3.13
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook