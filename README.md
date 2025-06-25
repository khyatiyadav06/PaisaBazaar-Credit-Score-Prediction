# PaisaBazaar-Credit-Score-Prediction using Machine Learning

This project aims to build a robust machine learning model to classify customer credit scores into three categories: **Poor**, **Standard**, and **Good**, using financial and behavioral features. The final deployed model is based on **XGBoost**, offering high accuracy and business interpretability.

---

## 📁 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Model Explainability](#model-explainability)
- [Deployment Preparation](#deployment-preparation)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [License](#license)

---

## 📌 Overview

A credit score prediction system that leverages customer financial data to assess their creditworthiness using machine learning. This can help financial institutions minimize loan default risks and make faster, data-driven decisions.

---

## 🧾 Dataset

- Format: `.csv`
- Records: 100,000 customer data entries
- Features include:
  - Demographics: Age, Occupation
  - Financial: Annual Income, Monthly Salary, EMI, Debt, Credit History
  - Behavioral: Delayed Payments, Credit Inquiries, Credit Mix, etc.
- Target Variable: `Credit_Score` (Poor = 0, Standard = 1, Good = 2)

---

## 🧠 Problem Statement

To develop a supervised machine learning model that predicts the **Credit Score category** of a customer based on historical financial behavior and profile features.

---

## 🔄 Project Pipeline

1. **Data Cleaning & Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Selection**
4. **Model Training & Evaluation**
5. **Model Explainability**
6. **Model Saving for Deployment**

---

## 📊 Exploratory Data Analysis

- Univariate and bivariate plots to explore distributions and relationships
- Multivariate analysis to understand key influencing factors
- Insights visualized using `Seaborn`, `Matplotlib`

---

## 🛠️ Feature Engineering

- Handled missing values
- Categorical encoding using Label Encoding and One-Hot Encoding
- Feature scaling with `StandardScaler`
- Outlier treatment using IQR method
- Feature selection using correlation, tree-based methods, and variance analysis

---

## 🤖 Model Building

### Models Trained:
- Random Forest Classifier
- XGBoost Classifier ✅ (Selected)

### Final Model:
```python
XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
