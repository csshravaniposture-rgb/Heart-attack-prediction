import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("dt_model (1).pkl")

st.title("Heart Disease Prediction App")

# Inputs
age = st.number_input("Age")
gender = st.number_input("Gender (0 = Female, 1 = Male)")
heart_rate = st.number_input("Heart rate")
systolic_bp = st.number_input("Systolic blood pressure")
diastolic_bp = st.number_input("Diastolic blood pressure")
blood_sugar = st.number_input("Blood sugar")
ck_mb = st.number_input("CK-MB")
troponin = st.number_input("Troponin")

# Create dataframe (Column names MUST match exactly)
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Heart rate": [heart_rate],
    "Systolic blood pressure": [systolic_bp],
    "Diastolic blood pressure": [diastolic_bp],
    "Blood sugar": [blood_sugar],
    "CK-MB": [ck_mb],
    "Troponin": [troponin]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction Result: {prediction}")
