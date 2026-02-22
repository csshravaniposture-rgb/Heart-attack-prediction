import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("dtc_model.pkl")

st.title("HEART ATTACK PREDICTION")

Age = st.number_input("Age", min_value=1, max_value=120)
Gender = st.selectbox("Gender", ["Male", "Female"])
Heart_rate = st.number_input("Heart Rate")
Systolic_blood_pressure = st.number_input("Systolic Blood Pressure")
Diastolic_blood_pressure = st.number_input("Diastolic Blood Pressure")
Blood_sugar = st.number_input("Blood Sugar")
Troponin = st.number_input("Troponin")

Gender = 1 if Gender == "Male" else 0

input_data = pd.DataFrame({
    "Age": [Age],
    "Gender": [Gender],
    "Heart_rate": [Heart_rate],
    "Systolic_blood_pressure": [Systolic_blood_pressure],
    "Diastolic_blood_pressure": [Diastolic_blood_pressure],
    "Blood_sugar": [Blood_sugar],
    "Troponin": [Troponin]
})

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error(" High Risk of Heart Attack")
        else:
            st.success(" Low Risk of Heart Attack")

    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Check if feature names and order match training data.")
