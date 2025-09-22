# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from src.optimizer import optimize_order
from src.features import create_lag_features

st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")

st.title("📦 Supply Chain Optimization Demo")

# --- Load data ---
if not os.path.exists("data/synthetic_sales.csv"):
    st.error("Run notebook first to generate data/synthetic_sales.csv")
    st.stop()

df = pd.read_csv("data/synthetic_sales.csv", parse_dates=["date"])
features_df = create_lag_features(df)

# --- Load or train model ---
if not os.path.exists("model/rf_demand.joblib"):
    st.warning("Model not found, training now...")
    X = features_df[[c for c in features_df.columns if c not in ["date","sku","sales"]]]
    y = features_df["sales"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_demand.joblib")
else:
    model = joblib.load("model/rf_demand.joblib")

# --- UI: pick SKU ---
sku = st.selectbox("Choose SKU", sorted(df["sku"].unique()))

sku_df = df[df["sku"]==sku].sort_values("date")
st.line_chart(sku_df.set_index("date")["sales"])

# --- Forecast next-day demand ---
latest = features_df[features_df["sku"]==sku].sort_values("date").tail(1)
latest_features = latest.drop(columns=["date","sku","sales"])
pred = round(model.predict(latest_features)[0])
st.metric("Predicted demand for tomorrow", f"{pred} units")

# --- Optimizer ---
on_hand_input = st.number_input("Current stock (on-hand)", min_value=0, value=50)
order = optimize_order({sku: pred}, {sku: on_hand_input}, lead_time_days=7,
                       holding_cost=0.05, shortage_cost=1.0)

st.success(f"Recommended order for {sku}: {order[sku]} units")
