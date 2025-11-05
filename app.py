import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Paths to model artifacts  
MODEL_PATH = "models/xgb_financial_health.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
CLASS_MAP_PATH = "models/class_mapping.pkl"

# Safety check to ensure model files exist
missing = [p for p in (MODEL_PATH, FEATURES_PATH, CLASS_MAP_PATH) if not os.path.exists(p)]
if missing:
    st.error(f"Model files are missing: {missing}. Please ensure all model files are in the 'models' directory.")
    st.stop()

# Load artifacts
model = joblib.load(MODEL_PATH)
features_column = joblib.load(FEATURES_PATH)   # expected: list of feature column names in correct order
class_mapping = joblib.load(CLASS_MAP_PATH)     # expected: mapping index -> label, e.g. {0: "Healthy", 1: "At Risk", 2: "Critical"}

# If class_mapping keys are strings, convert to int-key mapping if possible
# (Example: some code saves {"0": "Healthy", ...} — handle this)
if all(isinstance(k, str) and k.isdigit() for k in class_mapping.keys()):
    class_mapping = {int(k): v for k, v in class_mapping.items()}

# Streamlit page setup
st.set_page_config(page_title="SME Financial Health Monitor", layout="wide", initial_sidebar_state="expanded")

# Simple CSS (optional)
st.markdown("""
    <style>
    body { background-color: #0e1117; color: #fafafa; font-family: 'Segoe UI', sans-serif; }
    .title { text-align: center; font-size: 40px; color: #00ffb3; font-weight: bold; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #b3b3b3; font-size: 18px; margin-bottom: 30px; }
    .result-card { background-color: #1c1f26; padding: 25px; border-radius: 15px; text-align: center;
                   box-shadow: 0px 0px 10px rgba(0,255,179,0.4); transition: transform 0.3s ease; }
    div.stButton > button { background-color: #00ffb3; color: black; font-weight: bold; border-radius: 10px; width: 100%; height: 50px;}
    </style>
""", unsafe_allow_html=True)

st.title("AI-Driven Financial Health Monitor")
st.markdown("This application leverages advanced machine learning techniques to assess the financial health of businesses")
st.divider()

# Inputs
st.subheader("Enter financial ratios")
col1, col2, col3 = st.columns(3)
with col1:
    roa = st.number_input("Return on Assets (ROA)", value=0.06, step=0.01, format="%.4f")
    current_ratio = st.number_input("Current Ratio", value=1.5, step=0.1, format="%.2f")
with col2:
    roe = st.number_input("Return on Equity (ROE)", value=0.12, step=0.01, format="%.4f")
    debt_to_equity = st.number_input("Debt to Equity Ratio", value=1.0, step=0.1, format="%.2f")
with col3:
    gross_margin = st.number_input("Gross Margin", value=0.4, step=0.01, format="%.4f")

input_df = pd.DataFrame({
    'ROA': [roa],
    'Current_Ratio': [current_ratio],
    'ROE': [roe],
    'Debt_to_Equity': [debt_to_equity],
    'Gross_Margin': [gross_margin]
})
# Auto-align input_df columns to model's expected feature names 
expected = [str(f) for f in features_column] 

def normalize(name):
    # lowercase, remove spaces and underscores for fuzzy matching
    return "".join(name.lower().replace("_", " ").split())

input_map = { normalize(col): col for col in input_df.columns }
rename_dict = {}
for feat in expected:
    key = normalize(feat)
    if key in input_map:
        rename_dict[input_map[key]] = feat

if rename_dict:
    input_df = input_df.rename(columns=rename_dict)

# final safety check
missing_feats = [f for f in expected if f not in input_df.columns]
if missing_feats:
    st.error(f"After auto-alignment, missing required feature columns: {missing_feats}")
    st.stop()
st.divider()

# Prediction
if st.button("Predict Financial Health"):
    # ensure all required features exist in input_df
    missing_feats = [f for f in features_column if f not in input_df.columns]
    if missing_feats:
        st.error(f"Input is missing required feature columns: {missing_feats}")
    else:
        # Use only the features expected by the model (in the same order)
        X = input_df[features_column]

        # prediction index (e.g., 0,1,2)
        pred_index = int(model.predict(X)[0])

        # probabilities for each class (ordered by model.classes_)
        proba = model.predict_proba(X)[0]

        # Determine the position/index of pred_index in model.classes_ (to be safe)
        if hasattr(model, "classes_"):
            # model.classes_ gives class labels order
            # find index position in proba array that corresponds to pred_index
            try:
                pos = list(model.classes_).index(pred_index)
            except ValueError:
                pos = None
        else:
            pos = pred_index  # fallback if not available

        if pos is None:
            confidence = max(proba) * 100
        else:
            confidence = proba[pos] * 100

        # Map predicted index to human label
        status = class_mapping.get(pred_index, str(pred_index))

        color_map = {"Healthy": "green", "At Risk": "orange", "Critical": "red"}
        color = color_map.get(status, "white")

        st.markdown(f"### Financial Health Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence Level:** {confidence:.2f}%")
        st.progress(int(min(max(confidence, 0), 100)))

        # Visualization
        labels = [class_mapping.get(int(c), str(c)) for c in model.classes_]
        fig, ax = plt.subplots()
        bars = ax.bar(labels, proba)
        ax.set_ylabel('Probability')
        ax.set_title('Financial Health Probabilities')
        ax.set_ylim(0, 1)
        for bar, p in zip(bars, proba):
            height = bar.get_height()
            ax.annotate(f'{p:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        st.pyplot(fig)

        # Recommendations
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
        st.markdown("**Note:** These are algorithmic suggestions. Consult a financial advisor for a full analysis.")
        st.success("Recommendation generated successfully!")
