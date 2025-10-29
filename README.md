# Autism Spectrum Disorder (ASD) Prediction using Machine Learning

## Overview

This project focuses on building a machine learning model to predict whether an individual shows signs of Autism Spectrum Disorder (ASD) based on various behavioral and demographic attributes.
The dataset consists of screening responses collected from adults through a questionnaire, with the target variable being `Class/ASD` (0: No, 1: Yes).

---

## Tech Stack & Tools

* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imblearn`, `pickle`
* **Environment:** Google Colab
* **Version Control:** Git & GitHub

---

## Dataset

* **Filename:** `train.csv`
* The dataset contains features like age, gender, ethnicity, country of residence, and responses to 10 behavioral questions (`A1_Score` to `A10_Score`).
* The target column is `Class/ASD` (1 = ASD Positive, 0 = ASD Negative).

**Key preprocessing steps:**

* Dropped non-informative columns (`ID`, `age_desc`).
* Fixed inconsistent country names (`Viet Nam → Vietnam`, `AmericanSamoa → United States`, etc.).
* Replaced missing values in `ethnicity` and `relation` columns with `"Others"`.
* Converted categorical features to numerical values using **Label Encoding**.
* Handled outliers in numerical columns (`age`, `result`) by replacing them with median values.
* Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.

---

## Exploratory Data Analysis (EDA)

Performed detailed EDA to understand:

* **Distribution of numerical features** using histograms and box plots.
* **Categorical feature counts** using count plots.
* **Correlation heatmap** to detect relationships between features.

**Insights from EDA:**

* Some outliers in `age` and `result` columns.
* Noticeable class imbalance in the target variable (`Class/ASD`).
* No features showed strong multicollinearity.
* Ethnicity and relation columns required cleaning and normalization.

---

## Data Preprocessing

| Step              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| Outlier Treatment | Replaced extreme values with median using IQR method |
| Label Encoding    | Converted categorical columns to numeric             |
| Class Balancing   | Applied SMOTE to balance minority (ASD) class        |
| Train-Test Split  | 80:20 ratio using `train_test_split`                 |

---

## Model Training

Three tree-based models were trained and evaluated:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

Each model was tuned using **RandomizedSearchCV** with 5-fold cross-validation.

---

## Model Selection & Results

After tuning, the **Random Forest Classifier** achieved the best results.

```python
Best Model:
RandomForestClassifier(bootstrap=False, max_depth=20, n_estimators=50, random_state=42)
```

### 📈 Model Performance

| Metric                                 | Score    |
| -------------------------------------- | -------- |
| **Cross-Validation Accuracy (5-Fold)** | **0.93** |
| **Test Accuracy**                      | **0.82** |

**Confusion Matrix**

|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | 108          | 16            |
| **Actual Yes** | 13           | 23            |

**Classification Report**

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------- | ------ | -------- |
| 0 (No ASD)           | 0.89      | 0.87   | 0.88     |
| 1 (ASD)              | 0.59      | 0.64   | 0.61     |
| **Overall Accuracy** | **0.82**  |        |          |

---

## Key Insights

* **SMOTE** significantly improved recall for minority (ASD-positive) cases.
* The **Random Forest model** generalized well with strong test accuracy (0.82).
* The model demonstrates practical value for **early ASD screening**, especially for behavioral pattern-based datasets.

---

## 🧾 Files in Repository

| File                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `train.csv`               | Dataset used for model training                |
| `Autism_Prediction.ipynb` | Complete Colab notebook with code              |
| `best_model.pkl`          | Serialized best-performing Random Forest model |
| `encoders.pkl`            | Saved LabelEncoders for categorical variables  |
| `README.md`               | Project documentation                          |

---

## Author

**Maitri Jain**
B.Tech in Computer Science (5th Semester)
IIT Madras BS in Data Science (Foundation Level Completed)
