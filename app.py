import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.title("🩺 Breast Cancer Detection App")
st.write("Machine Learning project developed during CodeAlpha Internship")

# Load data
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = cancer_data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.success(f"Random Forest Accuracy: {accuracy:.4f}")

# User input section
st.subheader("🔍 Predict Cancer Type")

user_input = {}
for feature in cancer_data.feature_names:
    user_input[feature] = st.number_input(feature, float(X[feature].mean()))

input_df = pd.DataFrame([user_input])
input_scaled = sc.transform(input_df)

prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)

if st.button("Predict"):
    if prediction == 1:
        st.success("✅ Benign Tumor Detected")
    else:
        st.error("⚠️ Malignant Tumor Detected")

    st.write("Prediction Probability:", prediction_proba)
