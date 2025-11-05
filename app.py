import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import os

# load the model and metadata paths
model_path = "models/xgb_financial_health.pkl"
features_path = "models/feature_columns.pkl"
class_mapping_path = "models/class_mapping.pkl"

#safety check to ensure model files exist
if not os.path.exists(model_path) or not os.path.exists(features_path) or not os.path.exists(class_mapping_path):
    st.error("Model files are missing. Please ensure all model files are in the 'models' directory.")
    st.stop()

#load the model and related data
model = joblib.load(model_path)
features_column = joblib.load(features_path)
class_mapping = joblib.load(class_mapping_path)


# streamlit page setup
st.set_page_config(page_title="SME Financial Health Monitorr", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# css styling
st.markdown("""
             <style>
        /* Background and font setup */
        body {
            background-color: #0e1117;
            color: #fafafa;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Title styling */
        .title {
            text-align: center;
            font-size: 40px;
            color: #00ffb3;
            font-weight: bold;
            margin-bottom: 5px;
        }
        /* Subheader */
        .subtitle {
            text-align: center;
            color: #b3b3b3;
            font-size: 18px;
            margin-bottom: 30px;
        }
        /* Input container */
        .stNumberInput label {
            color: #ffffff;
            font-weight: 600;
        }
        /* Results card */
        .result-card {
            background-color: #1c1f26;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 0px 10px rgba(0,255,179,0.4);
            transition: transform 0.3s ease;
        }
        .result-card:hover {
            transform: scale(1.02);
        }
        /* Button style */
        div.stButton > button {
            background-color: #00ffb3;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            width: 100%;
            height: 50px;
        }
        div.stButton > button:hover {
            background-color: #00b37a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI-Driven Financial Health Monitor")   
st.markdown("This application leverages advanced machine learning techniques to assess the financial health of businesses")

st.divider()

#Financial ratios input
st.subheader("Enter financial Ratios")

col1, col2, col3 = st.columns(3)
with col1:
    roa = st.number_input("Return on Assets (ROA)", value=0.06, step=0.01)
    current_ratio = st.number_input("Current Ratio", value=1.5, step=0.1)
    
with col2:
    roe = st.number_input("Return on Equity (ROE)", value=0.12, step=0.01)
    debt_to_equity = st.number_input("Debt to Equity Ratio", value=1.0, step=0.1)

with col3:
    gross_margin = st.number_input("Gross Margin", value=0.4, step=0.01)

# Prepare input data for prediction
input_df = pd.DataFrame({
    'ROA': [roa],
    'Current_Ratio': [current_ratio],
    'ROE': [roe],
    'Debt_to_Equity': [debt_to_equity],
    'Gross_Margin': [gross_margin]
})

st.divider()

# Prediction button
if st.button("Predict Financial Health"):
    # predict clas and probabilities
    prediction = model.predict(input_df[features_column])
    proababilities = model.predict_proba(input_df)[0]

    status = class_mapping[prediction[0]]
    confidence = proababilities[prediction][0] * 100

    #color coding based on status
    color_map = {
        "Healthy": "green",
        "At Risk": "orange",
        "Critical": "red"}
    color = color_map.get(status, "black")
    
    st.markdown(f"### Financial Health Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.write(f"**Confidence Level:** {confidence:.2f}%")
    st.progress(int(confidence))

#visualization of probabilities
    fig, ax = plt.subplots()
    bars = ax.bar(class_mapping.values(), proababilities, color=['green', 'orange', 'red'])
    ax.set_ylabel('Probability')
    ax.set_title('Financial Health Probabilities')
    ax.set_ylim(0, 1)

    for bar, prob in zip(bars, proababilities):
        height = bar.get_height()
        ax.annotate(f'{prob:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

# AI recommendations
    st.markdown("AI-Driven Recommendations and insights")
    
    
    insights = []
    if roa < 0.05:
        insights.append("Low ROA - Improve efficiency in asset utilization.")
    else:
        insights.append("Strong ROA - Efficient asset performance.")

    if roe < 0.1:
        insights.append("Weak ROE - Increase profitability or manage equity better.")
    else:
        insights.append("Healthy ROE - Profitable shareholder returns.")

    if current_ratio < 1:
        insights.append("Low Liquidity - Risk of short-term insolvency. Improve working capital.")
    elif current_ratio > 3:
        insights.append("Excess Liquidity - Consider investing idle assets.")
    else:
        insights.append("Optimal Liquidity - Balanced short-term assets and liabilities.")

    if debt_to_equity > 2:
        insights.append("High Leverage - Too much debt. Consider deleveraging.")
    else:
        insights.append("Healthy Leverage - Good financial structure.")

    if gross_margin < 0.3:
        insights.append("Low Margin - Improve pricing or reduce production costs.")
    else:
        insights.append("Strong Margin - Good operational efficiency.")

    for tip in insights:
        st.markdown(f"- {tip}")

    st.markdown("**Note:** These insights are generated based on the input financial ratios and may not cover all aspects of financial health. Consult a financial advisor for comprehensive analysis.")
    
    st.success("recommendation Generated successfully!")
