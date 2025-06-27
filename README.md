# 🏦 Home Credit Default Risk Prediction

This project focuses on building a machine learning model to predict the likelihood of a client defaulting on a loan, using the **Home Credit Default Risk** dataset from Kaggle.

---

## 📌 Objective

To develop an end-to-end classification pipeline that:
- Handles missing values and categorical variables
- Performs feature selection using LightGBM and RFECV
- Trains a robust model using LightGBM and/or XGBoost
- Evaluates model performance using AUC-ROC

---

## 📂 Dataset Used

- `application_train.csv`: Training data with client information and target labels
- `application_test.csv`: Test data without the target
- **Note:** Other supporting files like `bureau.csv`, `previous_application.csv`, etc., are acknowledged but not used in this version.

---

## 🔧 Project Pipeline

### 1. **Data Preprocessing**
- Loaded `application_train.csv` and `application_test.csv`
- Identified and handled anomalies:
  - `DAYS_EMPLOYED = 365243` was treated as missing
  - Age calculated from `DAYS_BIRTH`
- Missing values imputed:
  - Numerical features filled with `-999`
  - Categorical features filled with `'missing'`
- One-hot encoding and label encoding were used appropriately:
  - `get_dummies()` used for LightGBM/XGBoost compatibility
  - Columns aligned between train and test to ensure consistent features

### 2. **Feature Selection**
- Used `LightGBMClassifier` with `RFECV` and `StratifiedKFold` to select optimal features
- Feature selection scoring based on `roc_auc`
- Final model used only the top `82` selected features

### 3. **Model Building**
- LightGBM used for final classification and Stratify for dealing with Imbalanced dataset.
- Early stopping and evaluation metric (`AUC`) were applied
- Column names cleaned to avoid JSON-related errors in LightGBM

### 4. **Final Output**
- Predictions made on the test set
- Achieved ROC-AUC score of 0.76
- Output prepared in the required Kaggle submission format

---

## 📈 Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn` (RandomForest, RFECV, preprocessing, imputation)
- `lightgbm`, `xgboost`

---

## 📊 Evaluation Metric

- **ROC-AUC**: Used during cross-validation and model tuning
- Stratified K-Fold used to maintain class distribution during validation

---

## 🧠 Key Learnings

- Effectively handled missing data in both numeric and categorical features
- Understood how data leakage can occur if missing values are imputed before train-test split using learned statistics (like mean or median). I avoided this by imputing with placeholder values (e.g., -999 or 'missing'), which do not learn from the data distribution, and then aligned and split the data safely.
- Learned to work with large-scale datasets and manage memory and performance
- Gained insight into feature selection using `RFECV` for dimensionality reduction
- Learned best practices for aligning train/test sets after one-hot encoding

---
---

### 📊 Power BI Dashboard – Interactive Loan Default Risk Analysis

To complement the machine learning model, an **interactive Power BI dashboard** was developed to uncover business-level insights and interpret model predictions visually.

#### ✅ Dashboard Highlights

Risk Segmentation
Visualized applicant risk levels by bucketing prediction probabilities into four categories:

Low Risk (0.00–0.20)

Medium Risk (0.21–0.50)

High Risk (0.51–0.90)

Default Likely (>0.90)


Demographic & Employment Trends
Custom fields like AGE and EMPLOYMENT_YEARS were derived from raw date fields to analyze patterns across age groups and job duration.



* **Risk Category Slicer & Customer Lookup**
  Enables filtering all visuals based on predicted `Risk Category` and specific `Customer ID`, making exploration tailored and intuitive.

* **Donut Chart – Client Companion Analysis**
  Displays who accompanied the client during the loan application (e.g., spouse, friend), offering insights into behavioral factors affecting risk.

* **Stacked Bar Charts**

  * **Number of Children vs Income Type:** Understand how family structure and income source relate to default probability.
  * **Contract Type vs Risk Level:** Examine how different loan types (e.g., Cash loans, Revolving loans) correlate with applicant risk.

* **Pie Charts**

  * **Realty Ownership vs Risk:** Visualizes how owning real estate impacts default risk.
  * **Car Ownership vs Risk:** Compares default tendencies of applicants owning vs not owning a car.

* **Treemap – Occupation vs Loan Amount**
  Shows which occupational groups kept the highest amounts to secure their loans.

* **Bar Chart – Organization Type vs Income**
  Maps different organization types to the total income reported by applicants.

* **Funnel Chart – Total Income by Age Group**
  Illustrates income distribution across age buckets to identify income trends in high-risk demographics.

* **Clustered Column Chart – Age, Employment Years, and Risk**
  Analyzes how applicant age and employment duration influence loan default probability.

#### 🎯 Business Value

The dashboard empowers business stakeholders to:

* Identify key applicant traits driving loan defaults.
* Compare risk across demographic, behavioral, and financial segments.
* Gain trust in the ML model by visualizing its predictions in a business-friendly format.

---

## 🚀 Future Work

- Integrate additional files like `bureau.csv`, `installments_payments.csv`, etc., to engineer richer features
- Explore more advanced imputation techniques (`KNNImputer`, `IterativeImputer`)
- Use model interpretation techniques like SHAP for explainability
- Deploy the final model using Streamlit or Flask for user-friendly prediction

---

## 🏁 Acknowledgments

- [Kaggle: Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- Home Credit Group for providing real-world financial datasets

---
