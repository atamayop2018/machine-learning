# Machine Learning Analysis Report

## Overview

This project addresses a binary classification problem: predicting the presence of heart disease in patients based on clinical diagnostic features. The dataset used is the **Heart Disease** dataset from the UCI Machine Learning Repository (Janosi et al., 1988), which contains 303 patient records with 13 clinical attributes collected from the Cleveland Clinic Foundation. A Random Forest Classifier was selected as the primary model after comparison against a Logistic Regression baseline, and both models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. The dataset is publicly available at [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Dataset Description

The Heart Disease dataset represents clinical records of 303 patients who underwent diagnostic testing for cardiovascular disease at the Cleveland Clinic Foundation. After removing 6 rows with missing values, 297 records were used for modeling. The dataset contains 13 predictor features and 1 binary target variable (`HeartDisease`): age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiogram results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise (oldpeak), slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy (CA), and thalassemia type (Thal). The target variable indicates the presence (1) or absence (0) of heart disease, with a reasonably balanced distribution of 54% negative and 46% positive cases.

## Modeling Approach

### Data Preparation

Data preprocessing included three key steps. First, 6 rows containing missing values in the `CA` and `Thal` columns were removed, as the small number of missing records (2% of the dataset) did not warrant imputation, which could introduce its own biases (Hastie et al., 2009). Second, the data was split into training (80%) and test (20%) sets using stratified sampling to preserve the class distribution in both subsets. Third, all features were standardized using `StandardScaler`, fitting only on the training data to prevent data leakage — a common pitfall in machine learning pipelines (Kaufman et al., 2012).

### Model Selection

Two models were trained and compared:

1. **Logistic Regression** served as an interpretable baseline model. Logistic regression is well-suited for binary classification tasks and provides probabilistic outputs, making it a standard first model for medical prediction tasks (Hosmer et al., 2013).

2. **Random Forest Classifier** was selected as the primary model because ensemble methods generally achieve higher predictive performance on tabular data by combining multiple decision trees that capture nonlinear relationships and feature interactions (Breiman, 2001). Hyperparameters were tuned using `GridSearchCV` with 5-fold cross-validation over a grid of `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. The best configuration was: 100 estimators, max depth of 5, minimum samples split of 5, and minimum samples leaf of 1.

### Evaluation Metrics

Multiple classification metrics were used to provide a comprehensive evaluation:

- **Accuracy** measures overall correctness but can be misleading with imbalanced classes (Sokolova & Lapalme, 2009).
- **Precision** measures the proportion of positive predictions that are truly positive — important in medical contexts to minimize false alarms.
- **Recall (Sensitivity)** measures the proportion of actual positives correctly identified — critical for heart disease detection where missing a true case can be life-threatening.
- **F1-score** provides a harmonic mean of precision and recall, balancing both concerns (Sokolova & Lapalme, 2009).
- **ROC-AUC** evaluates the model's ability to discriminate between classes across all decision thresholds.

### Model Assumptions

Logistic regression assumes a linear relationship between features and the log-odds of the target, independence of observations, and no severe multicollinearity. The Random Forest has fewer assumptions about feature distributions but assumes that the training data is representative of the deployment population. Both models assume that the features available are informative for the prediction task.

## Results

The Random Forest model outperformed Logistic Regression on most metrics when evaluated on the held-out test set (60 samples):

| Metric     | Logistic Regression | Random Forest |
|------------|---------------------|---------------|
| Accuracy   | 0.833               | 0.850         |
| Precision  | 0.846               | 0.880         |
| Recall     | 0.786               | 0.786         |
| F1 Score   | 0.815               | 0.830         |
| ROC-AUC    | 0.950               | 0.951         |

The Random Forest achieved an accuracy of 85.0%, an F1 score of 0.830, and a ROC-AUC of 0.951 on the test set. The confusion matrix showed that the model correctly classified 29 out of 32 healthy patients and 22 out of 28 patients with heart disease. The 6 false negatives (patients with disease classified as healthy) represent the most clinically concerning errors.

Cross-validation on the training set yielded a mean F1 score of 0.810 (±0.107) for the Random Forest, indicating moderate variability across folds — expected given the small dataset size.

Feature importance analysis revealed that **ChestPainType** (0.173), **Thal** (0.140), **CA** (0.121), **MaxHR** (0.118), and **Oldpeak** (0.105) were the top five most predictive features, consistent with clinical knowledge about heart disease risk factors.

## Interpretation for a Non-Technical Audience

The goal of this project was to build a computer program that can help predict whether a patient might have heart disease based on medical test results. Think of it like a checklist that looks at a patient's age, blood pressure, cholesterol, heart rate during exercise, and several other medical measurements, and then makes a prediction.

Our best model correctly identified whether a patient had heart disease about 85% of the time. When the model predicted that a patient *did* have heart disease, it was right about 88% of the time (high precision). However, it missed about 21% of patients who actually had the disease (6 out of 28 patients were not flagged). In a medical setting, this means the model is useful as a screening aid, but it should not replace a doctor's judgment — some patients with disease could still be missed.

The model found that the type of chest pain a patient experiences, the results of a thalassemia blood test, and the number of major blood vessels visible on imaging were the strongest indicators of heart disease. These findings align with what doctors already know about heart disease diagnosis, which gives confidence that the model is learning meaningful patterns rather than random noise.

## Limitations and Potential Bias

### Limitations

1. **Small dataset size.** With only 297 usable records, the model has limited data to learn from, and the test set of 60 samples provides a noisy estimate of true performance. The cross-validation standard deviation of ±0.107 on F1 score reflects this instability. A larger dataset would improve both model performance and evaluation reliability (Vabalas et al., 2019).

2. **Single-source data.** The dataset was collected from a single medical center (Cleveland Clinic Foundation) in the 1980s. Medical practices, diagnostic criteria, and patient populations have changed significantly since then. The model may not generalize well to patients from different geographic regions, hospitals, or time periods.

3. **Feature limitations.** The dataset contains only 13 clinical features. Modern heart disease prediction often incorporates additional biomarkers, imaging data, genetic factors, and patient history that could improve prediction accuracy.

### Potential Bias

1. **Demographic bias.** The dataset is predominantly male (approximately 68% of records), which means the model may perform worse for female patients. Heart disease presents differently in women, and underrepresentation could lead to lower recall for female patients (Norris et al., 2020). This is a well-documented issue in medical AI systems.

2. **Selection bias.** The patients in this dataset were referred for cardiac catheterization at a specialized center, meaning they likely had higher clinical suspicion of disease than the general population. This referral bias means the model may not perform as well as a general screening tool for asymptomatic populations.

To mitigate these risks, this model should be used only as a decision-support tool alongside clinical judgment, not as a standalone diagnostic system. Future work should validate on diverse, contemporary datasets and evaluate model fairness across demographic subgroups.

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579

Norris, C. M., Yip, C. Y. Y., Engert, J. C., et al. (2020). Sex and gender in cardiovascular medicine. *Canadian Journal of Cardiology*, 36(7), S3–S12.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A. J. (2019). Machine learning algorithm validation with a limited sample size. *PLoS ONE*, 14(11), e0224365. https://doi.org/10.1371/journal.pone.0224365
