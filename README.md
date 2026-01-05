🩺 Breast Cancer Detection – CodeAlpha Internship Project
📌 Project Overview

This project focuses on Breast Cancer Detection using Machine Learning classification algorithms. It was developed as part of my CodeAlpha Internship. The goal of the project is to accurately classify tumors as malignant or benign using features extracted from breast cancer diagnostic data.

The project uses the Breast Cancer dataset from sklearn.datasets, performs data analysis, visualization, preprocessing, and applies multiple machine learning models to compare performance.

📂 Dataset

Source: sklearn.datasets.load_breast_cancer

Total Features: 30 numerical features

Target Variable:

0 → Malignant

1 → Benign

🛠️ Technologies & Libraries Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

📊 Exploratory Data Analysis (EDA)

The following visualizations were performed:

Pair plots for selected mean features

Heatmap of the dataset

Correlation matrix heatmap for feature relationships

Statistical summary and null-value checks

⚙️ Data Preprocessing

Converted dataset into a Pandas DataFrame

Checked for missing values (none found)

Split the dataset into training (80%) and testing (20%)

Applied StandardScaler for feature scaling

🤖 Machine Learning Models Implemented

The following classification algorithms were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Support Vector Classifier (SVC)

✅ Model Evaluation

Models were evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

⭐ Best Performing Model

Random Forest Classifier

Accuracy: 96.49%

Accuracy Score: 0.9649122807017544

This model achieved the highest accuracy among all tested classifiers.

📈 Results Summary
Model	Accuracy
Logistic Regression	High
K-Nearest Neighbors	High
Random Forest	96.49% (Best)
Support Vector Machine	High
🚀 Conclusion

The project demonstrates that machine learning models, especially Random Forest, can effectively classify breast cancer tumors with high accuracy. Such systems can assist medical professionals in early diagnosis and decision-making.

📌 Internship Information

Internship Company: CodeAlpha

Project Type: Machine Learning / Classification

Project Title: Breast Cancer Detection

👤 Author

Mudasir Iqbal
Machine Learning Intern – CodeAlpha
