import streamlit as st
import joblib
import numpy as np

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Predictor")

# Create input fields dynamically
input_data = []

for col in columns:
    value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    data = np.array([input_data])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")