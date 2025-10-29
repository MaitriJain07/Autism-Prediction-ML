# Autism Prediction using Machine Learning

This project aims to predict the likelihood of Autism Spectrum Disorder (ASD) based on screening responses and demographic data.
It applies data preprocessing, exploratory data analysis (EDA), and multiple classification models (Decision Tree, Random Forest, and XGBoost) with hyperparameter optimization to identify the most accurate predictive model.

---

## Project Overview

* **Goal:** To develop a machine learning model that can predict potential Autism Spectrum Disorder (ASD) traits from screening data.
* **Techniques Used:**

  * Exploratory Data Analysis (EDA)
  * Label Encoding
  * Outlier Detection and Handling
  * Class Imbalance Correction using SMOTE
  * Model Training & Evaluation
  * Hyperparameter Tuning using RandomizedSearchCV
* **Algorithms:**

  * Decision Tree
  * Random Forest
  * XGBoost

---

## Dataset

**Name:** Autism Screening Adult Data Set
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)
**Description:**
The dataset contains behavioral and personal attributes of adults who took an autism screening test.
It includes 20+ features like:

* `A1_Score` to `A10_Score` — screening question responses
* `age`, `gender`, `ethnicity`, `jaundice`, `autism`, `country_of_res`, `used_app_before`, `relation`
* **Target:** `Class/ASD` (1 = likely ASD traits, 0 = unlikely)

 *Note:* The dataset is renamed as `train_data.csv` for local use.

---

## ⚙️ Steps Performed

### 1. Data Understanding & Cleaning

* Handled missing values in `ethnicity` and `relation` columns.
* Merged rare categories under “Others”.
* Fixed inconsistent country names.
* Dropped redundant columns (`ID`, `age_desc`).

### 2. Exploratory Data Analysis (EDA)

* Plotted distributions for numerical features (`age`, `result`).
* Count plots for categorical variables.
* Correlation heatmap to check feature relationships.

### 3. Data Preprocessing

* Replaced outliers using the IQR method (median imputation).
* Encoded categorical features using Label Encoding.
* Balanced dataset using **SMOTE (Synthetic Minority Oversampling Technique)**.

### 4. Model Training

Trained three baseline classifiers:

* Decision Tree
* Random Forest
* XGBoost

Used **5-Fold Cross Validation** to evaluate performance.

### 5. Hyperparameter Tuning

Optimized model parameters using **RandomizedSearchCV** for all three algorithms.
Saved the best-performing model as `best_model.pkl`.

### 6. Model Evaluation

Evaluated on unseen test data using:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## Results

| Model         | Cross-Validation Accuracy |
| ------------- | ------------------------- |
| Decision Tree | ~88%                      |
| Random Forest | ~92%                      |
| XGBoost       | **~94% (Best Model)**     |

The **XGBoost model** achieved the highest accuracy and was selected as the final model.

---

## Files in this Repository

| File                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| `Autism_Prediction.ipynb` | Complete implementation notebook              |
| `train_data.csv`          | Dataset (UCI Autism Screening Adult Data Set) |
| `encoders.pkl`            | Saved label encoders                          |
| `best_model.pkl`          | Trained best model (XGBoost)                  |
| `README.md`               | Project documentation                         |

---

##  Future Work

* Build a Flask web app for real-time predictions.
* Experiment with deep learning models.
* Add explainability using SHAP or LIME.

---

##  Author

**Maitri Jain**
B.Tech in Computer Science | IIT Madras BS in Data Science
