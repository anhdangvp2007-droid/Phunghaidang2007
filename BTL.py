import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD PIPELINE
model = joblib.load("models/pipeline.pkl")

st.title("🏠 Vietnam House Price Prediction")

address = st.text_input("Address")
area = st.number_input("Area", 50, 500)
frontage = st.number_input("Frontage", 0, 20)
road = st.number_input("Access Road", 0, 20)
floors = st.number_input("Floors", 1, 10)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Address": address,
        "Area": area,
        "Frontage": frontage,
        "Access Road": road,
        "Floors": floors,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms
    }])
    pred_log = model.predict(input_df)
    pred = np.expm1(pred_log)

    adjusted_pred = pred / 0.65

    st.success(
        f"🏠 Predicted price: {adjusted_pred[0]:.2f} billion VND"
    )