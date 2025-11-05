import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal
from joblib import load
# load the saved model
model = load("model.joblib")

# streamlit page setup
st.set_page_config(page_title="SME Financial Health Monitorr", 
                   layout="wide",
                   initial_sidebar_state="expanded")
# custom css
st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff;
            padding: 1.5rem;
            border-radius: 12px;
        }
        h1, h2, h3 {
            color: #1f4e79;
        }
        .stButton>button {
            background-color: #1f4e79;
            color: white;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #274c77;
        }
        .metric-container {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)
# Page Header
st.title("💼 AI-Driven SME Financial Health Monitor")
st.markdown("Welcome to the AI Financial Health Monitoring App this tool uses a trained Z-Score model to analyze your company’s financial stability and provide early risk detection.Upload your financial data below to get started!")

#sidebar inputs
st.sidebar.header("Upload Financial Data")
input_method = st.sidebar.radio("Select Input Method:", ("Upload file", "Manual Entry"))


total_assets = st.sidebar.number_input("Total Assets", min_value=0.0, step=1000.0)
total_liabilities = st.sidebar.number_input("Total Liabilities", min_value=0.0, step=1000.0)
retained_earnings = st.sidebar.number_input("Retained Earnings", min_value=0.0, step=1000.0)
ebit = st.sidebar.number_input("EBIT (Earnings Before Interest & Tax)", min_value=0.0, step=1000.0)
market_value_equity = st.sidebar.number_input("Market Value of Equity", min_value=0.0, step=1000.0)
sales = st.sidebar.number_input("Sales (Revenue)", min_value=0.0, step=1000.0)

# prediction button
if st.sidebar.button("🔍 Predict Financial Health"):
 try:
        # Prepare input data
        input_data = np.array([[total_assets, total_liabilities, retained_earnings, ebit, market_value_equity, sales]])
        
        # Predict Z-score
        z_score = model.predict(input_data)[0]
        
        # Display results
        st.subheader("Prediction Result")
        st.markdown(f"Predicted Z-Score: {z_score:.3f}")

        # Interpretation block
        if z_score > 2.99:
            st.success("Financially Healthy — Low bankruptcy risk.")
            status = "Healthy"
            color = "green"
        elif 1.81 < z_score <= 2.99:
            st.warning("Grey Zone — Moderate financial risk. Monitor closely.")
            status = "Caution"
            color = "orange"
        else:
            st.error("Distressed Zone — High risk of bankruptcy.")
            status = "Distressed"
            color = "red"
 except Exception as e:
    st.error(f"Error during prediction: {e}")