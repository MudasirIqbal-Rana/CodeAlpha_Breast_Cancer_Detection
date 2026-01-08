import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.title("🩺 Breast Cancer Detection App")
st.write("Machine Learning project developed during CodeAlpha Internship")

@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y, data.feature_names

X, y, features = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("📊 Model Performance")
st.success(f"Random Forest Accuracy: {accuracy:.4f}")

st.subheader("🔍 Predict Cancer Type")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        feature, value=float(X[feature].mean())
    )

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)

    if prediction == 1:
        st.success("✅ Benign Tumor Detected")
    else:
        st.error("⚠️ Malignant Tumor Detected")

    st.write("Prediction Probability:", probability)
