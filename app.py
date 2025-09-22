import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Title
st.title("Supply Chain Optimization - Demand Forecasting")

# --- Interactive Inputs ---
stock_levels = st.number_input("Enter Stock Levels", min_value=0, max_value=10000, value=500)
lead_times = st.number_input("Enter Lead Time (days)", min_value=1, max_value=60, value=10)
price = st.number_input("Enter Price", min_value=1, max_value=1000, value=100)

# Put inputs in a dataframe (same structure as training features)
input_data = pd.DataFrame({
    "Stock levels": [stock_levels],
    "Lead times": [lead_times],
    "Price": [price]
})

# --- Load model (simple example: train inside app) ---
# Normally you'd load a pre-trained model
X = input_data  # placeholder, replace with your training X
y = [200]       # placeholder
model = LinearRegression()
model.fit(X, y)

# --- Predict ---
if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"📦 Predicted Demand for Next Week: {prediction:.0f} units")
