# Customer Churn Prediction – End-to-End ML Pipeline

## Objective
The objective of this project is to build a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The project demonstrates data preprocessing, model training, hyperparameter tuning, evaluation, and model export using Scikit-learn Pipeline.

---

## Dataset
**Dataset Used:** Telco Customer Churn Dataset  

The dataset contains customer demographic information, account details, services subscribed, and whether the customer churned or not.

**Target Variable:**  
- `Churn` (Yes/No → Converted to 1/0)

**Features Include:**  
- Customer demographics (gender, senior citizen, partner, dependents)  
- Account information (tenure, contract, payment method, monthly charges, total charges)  
- Services (internet service, phone service, streaming, etc.)  

---

## Methodology / Approach

### 1. Data Preprocessing
- Removed unnecessary columns such as `customerID`
- Converted `TotalCharges` to numeric
- Handled missing values using **SimpleImputer**
- Scaled numerical features using **StandardScaler**
- Encoded categorical variables using **OneHotEncoder**
- Used **ColumnTransformer** to apply transformations

### 2. Machine Learning Pipeline
Two models were implemented using Scikit-learn Pipeline:
- Logistic Regression
- Random Forest Classifier

### 3. Hyperparameter Tuning
Used **GridSearchCV** to tune model parameters:
- Logistic Regression → Regularization parameter C and solver
- Random Forest → Number of trees, max depth, min samples split, min samples leaf

### 4. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

### 5. Model Export
The best-performing model pipeline (including preprocessing) was saved using **joblib** for reuse in production.

---

## Results / Key Findings
- Both Logistic Regression and Random Forest performed well for churn prediction.
- Random Forest performed slightly better due to its ability to capture non-linear relationships.
- F1-score was used as the main evaluation metric because churn datasets are usually imbalanced.
- The final pipeline can be reused directly on new customer data for churn prediction.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## How to Run the Project

### 1. Install required libraries
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Run the Jupyter Notebook
```
jupyter notebook churn_pipeline.ipynb
```

### 3. Train the model and save the pipeline

### 4. Load the saved pipeline and make predictions
```python
import joblib
model = joblib.load("random_forest_pipeline.pkl")
predictions = model.predict(new_data)
```

---

## Skills Demonstrated
- End-to-end Machine Learning Pipeline
- Data Preprocessing and Feature Engineering
- Hyperparameter Tuning (GridSearchCV)
- Model Evaluation Metrics
- Model Export and Reusability
- Production-ready Pipeline Development

---

## Conclusion
This project demonstrates how to build a complete and reusable machine learning pipeline for customer churn prediction. The pipeline includes preprocessing, model training, evaluation, and export, making it suitable for real-world deployment scenarios.
