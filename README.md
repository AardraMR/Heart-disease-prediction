# Heart Disease Prediction using Machine Learning

**Author:** Aardra M R  
**Course:** AI With Python  
**Date:** May 16, 2026  

## 🚀 Project Overview
This project focuses on analyzing clinical and demographic patient data to build a predictive machine learning model for identifying the presence of heart disease. Early detection of cardiovascular vulnerabilities plays a crucial role in preventive healthcare and clinical decision-making.

The underlying dataset contains various key medical indicators including age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), and maximum heart rate achieved (thalach).

## 📁 Repository Structure
* **`heart.ipynb`**: Complete Jupyter Notebook containing the full data pipeline and model implementation.
* **`heart.csv`**: Structured dataset containing patient clinical features.
* **`Heart_disease_prediction_Report.pdf`**: Project documentation outlining methodologies and clinical insights.

## 🛠️ Technical Workflow & Implementation

### 1. Data Preprocessing
* Checked for missing values across all features; the dataset was confirmed clean with zero null values.
* Split data into target arrays:
  * **Features ($X$)**: All diagnostic/demographic indicators except the target.
  * **Target ($y$)**: Binary indicator for heart disease presence (0 or 1).
* Split the dataset using an 80/20 train-test ratio, resulting in:
  * **Training Size**: 820 samples
  * **Testing Size**: 205 samples

### 2. Exploratory Data Analysis (EDA)
* Analyzed distribution profiles for age metrics and cholesterol levels utilizing histograms and boxplots.
* Generated a full numerical correlation matrix to trace linear relationships between demographic variables, physical diagnostics, and ultimate target status.

### 3. Model Training
* **Algorithm Selected**: Logistic Regression (`max_iter=1000`)
* **Rationale**: Simple, highly efficient, and mathematically robust for binary classification on structured clinical tabular data.

## 📊 Evaluation & Performance Results
The model was tested on an unbiased test partition (205 samples) yielding the following classification metrics:

* **Overall Accuracy**: 79.51%
* **Confusion Matrix**:
  ```text
  [[73  29]  --> [True Negatives,  False Positives]
   [13  90]] --> [False Negatives, True Positives]