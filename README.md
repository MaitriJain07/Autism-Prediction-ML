# Autism Prediction using Machine Learning

This project builds a machine learning model to predict the likelihood of Autism Spectrum Disorder (ASD) using behavioral and screening data. The model analyzes user responses and key clinical indicators to assist in early screening and decision support.

---

##  Project Overview
- **Objective:** Predict Autism diagnosis using ML classification techniques.  
- **Tools Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab  
- **Dataset:** Autism Screening Adult Dataset (UCI)  

---

## ⚙️ Methodology
1. **Data Preprocessing** – Handling missing values, encoding categorical data  
2. **Exploratory Data Analysis (EDA)** – Visualizing correlations and feature distributions  
3. **Model Building** – Logistic Regression, Random Forest, and SVM models trained  
4. **Evaluation** – Accuracy, Precision, Recall, F1-Score, and ROC curve used for evaluation  
5. **Result** – Random Forest achieved the best accuracy (~94%)  

---

##  Key Insights
- Certain behavioral questions strongly correlate with ASD diagnosis  
- Feature selection and proper encoding significantly improved performance  
- Model generalizes well across demographic subgroups  

---

##  Future Improvements
- Integrate explainable AI (SHAP values) to understand model reasoning  
- Deploy model with Flask or Streamlit for interactive screening  
- Test model on larger real-world datasets  

---

##  Repository Contents
| File | Description |
|------|--------------|
| `autism_prediction.ipynb` | Full Google Colab notebook with code, output, and explanations |
| `train.csv` | Dataset used for training and testing the machine learning models. Contains screening responses and demographic attributes for ASD prediction. |
| `README.md` | Project overview and documentation |

---

##  Author
**Maitri Jain**  
B.S. Data Science (IIT Madras) | B.Tech in Computer Science  
*Project developed and executed in Google Colab for accessibility and reproducibility.*
