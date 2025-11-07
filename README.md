# 1. Business Understanding
# 1.1 Business Overview 
 Many investors, lenders, and business owners rely on intuition or outdated reports when evaluating a company's financial health, which often leads to poor investment or lending decisions.The project aims to develop a data-powered tool that automatically analyzes publicly available financial data (income statements, balance sheets, and cash flows) to assess a company’s financial stability, profitability, and risk.This project aims to build a scoring system that evaluates a company's financial health using real-world financial data.The project will simplify financial decision-making by transforming raw numbers into actionable insights through data analysis, visualization, and machine learning.Real financial datasets will be fetched directly from the Yahoo Finance API

## 1.2 Problem Statement

Many investors, lenders, and business owners rely on intuition or outdated reports when evaluating a company’s financial position. This lack of real-time, data-driven analysis can lead to poor investment or lending decisions.

Our challenge is to develop a **data-powered tool** that automatically analyzes publicly available financial data (like income statements, balance sheets, and cash flows) to assess a company’s **financial stability, profitability, and risk**.

This project will simplify financial decision-making by transforming raw numbers into actionable insights through **data analysis, visualization, and machine learning**.

---

## 1.3 Business Objectives

### Main Objective

To build a **data analysis and scoring system** that evaluates a company’s financial health using real-world financial data.

### Specific Objectives

1. To collect and preprocess financial data from **Yahoo Finance API**  
2. To analyze key financial metrics such as revenue growth, net income, debt-to-equity ratio, and cash flow trends.  
3. To build a **financial health scoring model** that assigns a score to each company based on performance indicators.  
4. To visualize financial insights using clear dashboards and charts for easier interpretation.  
5. To provide actionable recommendations for investors or business managers.

---

## 1.4 Research Questions

1. What financial indicators most accurately represent a company’s health and stability?  
2. How do profitability, liquidity, and leverage ratios correlate with a company’s risk level?  
3. Can we build a model that classifies companies into categories such as _Healthy_, _Moderate_, and _At Risk_?  
4. How can visualizing financial trends help investors make better decisions?

---

## 1.5 Success Criteria

- The system should accurately collect and clean financial data for multiple companies.  
- It should compute and visualize key financial ratios and trends.  
- The scoring model should produce realistic health scores based on financial fundamentals.  
- The final output should be clear and explainable to both technical and non-technical users.

---

# 2. Data Understanding

We will use **real financial datasets** fetched directly from APIs — not from Kaggle.

---

## Datasets & Sources

| Source | Type of Data | Description |
| --- | --- | --- |
| **Yahoo Finance API (via yfinance)** | Company financials | Income statements, balance sheets, cash flow, and stock history |
---

## Dataset Overview

Each company dataset will include:

- **Revenue**  
- **Gross profit**  
- **Operating income**  
- **Net income**  
- **Total assets & liabilities**  
- **Cash flow from operations**  
- **Debt-to-equity ratio**  
- **Return on assets (ROA)** and **Return on equity (ROE)**  
- **Stock price performance** over time  

These metrics help us assess profitability, liquidity, leverage, and efficiency — the four main pillars of financial health.

---

## Tools and Libraries

We’ll use the following tools for the analysis:

| Category | Libraries |
| --- | --- |
| **Data Collection** | `yfinance`, `requests`, `pandas` |
| **Data Cleaning & Processing** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Modeling & Scoring** | `scikit-learn`, `statsmodels` |
| **Deployment (Optional)** | `joblib` for model serialization |


# 3. Data Preparation

In this section, we will import the necessary Python libraries and load financial data directly from Yahoo Finance using the `yfinance` API. This will form the foundation of our analysis.

The data will include income statements, balance sheets, cash flow statements, and stock price history for a chosen company. We will then explore its structure before cleaning and feature engineering.


# Model Training

## Random Forest Model for Z-Risk Classification

This section loads the processed financial dataset, prepares predictor ratios, encodes the `Z_Risk` target, and trains a Random Forest classifier to predict company risk levels.

**Steps:**
1. Load `features.csv` into a DataFrame.
2. Define predictors: `ROA`, `ROE`, `Current Ratio`, `Debt to Equity`, `Gross Margin`.
3. Encode `Z_Risk` labels (`Safe=0`, `Grey=1`, `Distress=2`).
4. Split data into training and test sets (80/20).
5. Train a `RandomForestClassifier` with 200 trees.
6. Compute and print class weights for imbalance.
7. Evaluate model performance using `classification_report`.


## XGBoost Model for Z-Risk Prediction

This section trains an **XGBoost classifier** using financial ratios to predict `Z_Risk` categories and evaluates its performance.

**Steps:**
1. Generate `sample_weights` from computed class weights to handle imbalance.  
2. Initialize an `XGBClassifier` with tuned hyperparameters (300 trees, learning rate 0.05, depth 5).  
3. Fit the model on training data.  
4. Predict labels and probabilities on the test set.  
5. Evaluate results with:
   - `classification_report`
   - `confusion_matrix`
   - Overall accuracy score

## Section: Model Evaluation – Classification Report and Confusion Matrix

This section evaluates the trained model’s performance on the test dataset and visualizes prediction accuracy using a confusion matrix.

**Steps:**
1. Generate predictions (`y_pred`) and probabilities (`y_proba`) on test data.  
2. Print a detailed `classification_report` showing precision, recall, and F1-scores.  
3. Compute and display the `confusion_matrix` as a heatmap using Seaborn for clarity.

**Output:**  
- Console report of classification metrics.  
- Visual confusion matrix plot showing true vs. predicted classes.


## Model Evaluation – Classification Report and Confusion Matrix

This section evaluates the trained model’s performance on the test dataset and visualizes prediction accuracy using a confusion matrix.

**Steps:**
1. Generate predictions (`y_pred`) and probabilities (`y_proba`) on test data.  
2. Print a detailed `classification_report` showing precision, recall, and F1-scores.  
3. Compute and display the `confusion_matrix` as a heatmap using Seaborn for clarity.
    
![png](README_files/README_56_2.png)
    
## Cross-Validation for Model Stability

This section assesses the model’s robustness and consistency using **5-fold Stratified Cross-Validation**.

**Steps:**
1. Use `StratifiedKFold` to maintain class balance across folds.  
2. Evaluate the model with `cross_val_score` using accuracy as the metric.  
3. Print individual fold accuracies and summarize the mean and standard deviation.

**Output:**  
Displays per-fold accuracy scores and the overall mean ± standard deviation to gauge model stability.


     Cross-validation accuracy scores: [0.89108911 0.81188119 0.84158416 0.92       0.87      ]
    Mean CV Accuracy: 0.8669 ± 0.0376
    

## ROC-AUC Curve Analysis

Here we visualize the model’s discriminatory power using **ROC-AUC curves** for both binary and multiclass classification setups.

**Steps:**
1. Detect the number of target classes.  
2. For binary classification, plot a single ROC curve and compute the AUC score.  
3. For multiclass problems, binarize labels and plot per-class ROC curves, including a micro-average curve.  
4. Display the resulting ROC-AUC values to assess model performance across classes.

**Output:**  
- ROC curves plotted for binary or multiclass cases.  
- AUC scores printed per class (and overall micro-average).
   
![png](README_files/README_60_0.png)
    
    Per-class ROC-AUC: {0: 0.9700000000000001, 1: 0.8736434108527132, 2: 0.90625, 'micro': 0.9659347122831095}
    

## Precision–Recall Curve Analysis

This section evaluates model precision versus recall for both **binary** and **multiclass** classifications.

**Steps:**
1. If binary, compute and plot a single Precision–Recall curve with AUC.  
2. For multiclass, binarize labels and calculate per-class Precision–Recall (PR) curves and Average Precision (AP).  
3. Compute a **micro-average PR curve** to summarize overall model performance.  
4. Visualize all class curves and print per-class and overall AP scores.

**Output:**  
- PR curves for each class and micro-average curve.  
- Average Precision (AP) values displayed per class and overall.
 
![png](README_files/README_62_0.png)
    
    Per-class Average Precision (AP):
    Class 0: AP = 0.250
    Class 1: AP = 0.606
    Class 2: AP = 0.982
    Micro-average AP = 0.944
    

## Feature Importance Visualization

This block displays the **top 5 most important features** in the trained Random Forest model.  
It helps identify which financial ratios contribute most to Z-Risk classification.

**Steps:**
1. Extract and sort feature importances from the trained model.  
2. Visualize the top five features using a horizontal bar chart.  
3. Interpret higher bars as stronger influence on model predictions.
   
![png](README_files/README_64_0.png)
    
## SHAP Feature Contribution Analysis

This section uses **SHAP (SHapley Additive exPlanations)** to interpret the Random Forest model’s predictions.  
It identifies how each input feature influences the output across all classes.

**Steps:**
1. Initialize a SHAP TreeExplainer for the trained Random Forest model.  
2. Compute SHAP values for the test set.  
3. For multiclass cases, sum absolute SHAP values across classes to gauge overall impact.  
4. Display a bar-based summary plot of feature contributions.

**Output:**  
A SHAP summary bar plot highlighting the most influential features driving classification decisions.
   
![png](README_files/README_66_0.png)
    
## Environment Setup – Package Installation

This step ensures that all required libraries are installed for model persistence and web app deployment.

**Installed Packages:**
1. **joblib** – used for saving and loading trained models efficiently.  
2. **streamlit** – used for creating interactive web applications for visualizing and exploring results.

## Model Saving and Export

After training the XGBoost model, this block saves all essential components for deployment and later inference.

**Key Steps:**
- Creates a `models/` directory if it doesn’t exist.  
- Saves the trained **XGBoost model** using `joblib`.  
- Stores the **feature column list** required for future predictions.  
- Saves the **class mapping** for decoding model output labels.


    Model saved: models/xgb_financial_health.pkl
    Feature columns saved
    Class mapping saved
    

# Key Points & Summary of Work
The project successfully addressed the initial stages of the business objective: transforming raw financial data into actionable features for a scoring system.

* Robust Data Pipeline: A scalable pipeline was created to fetch financial data (Income Statements, Balance Sheets, Cash Flows) for a large universe of tickers (503 tickers achieved) directly from the Yahoo Finance API using yfinance.

* Data Standardization: A critical resolve_item_names function was implemented using fuzzy matching to reliably map inconsistent raw row names from the API (like "Total Revenues" or "Total Revenue") to standardized clean feature names.

Comprehensive Feature Set: Over 12 core financial features were engineered, covering the following pillars of financial health:

* Profitability Ratios: Gross Margin, Operating Margin, Net Margin, ROA, and ROE.

* Liquidity & Leverage Ratios: Current Ratio and Debt to Equity.

* Solvency Score: The Altman Z-Score was computed as a primary, multi-factor indicator of bankruptcy risk.

* Data Quality and Precision: All financial calculations utilized Python's Decimal type via a vec_safe_div function to ensure high arithmetic precision and prevent division-by-zero errors.

# Conclusions
Based on the data framework the following steps were done:

* Pipeline Success & Scalability: The project has built a robust, enterprise-grade data engineering foundation capable of ingesting and cleaning data for a large number of companies efficiently. This fulfills the requirement for a real-time, data-driven analysis tool.

* Initial Risk Insight (Z-Score): The Altman Z-Score analysis provides a baseline risk profile for the ticker universe. The statistics show a Mean Z-Score of 1.34 and a Median of 1.28. Since a Z-Score below 1.81 is classified as 'Distress', this suggests that, according to the Z-Score model, the average company in the sample exhibits significant financial distress risk.

* Data Quality Audit: The audit revealed a minimal number of unexpected negative values in fields like Gross Profit (1 instance) and Net Income (22 instances). The existence of these negative values is valid for loss-making companies but confirms the importance of the cleaning steps to prevent critical errors in ratio calculations.

# Recommendations
1. **For Safe Companies**:
Maintain current financial practices while exploring strategic reinvestment opportunities, such as expanding product lines or entering new markets to sustain growth momentum.

2. **For Grey Zone Companies**:
Conduct a cost structure and efficiency review to identify areas where expenses can be optimized. Strengthening liquidity reserves and improving revenue consistency should be prioritized.

3. **For Distressed Companies**:
Implement debt restructuring or refinancing strategies to reduce financial pressure. Immediately evaluate operational inefficiencies and consider divesting underperforming assets.

4. **For Investors & Lenders**:
Use the financial health score as a screening tool before deeper due diligence. Prioritize “Safe” companies, monitor “Grey” companies closely, and approach “Distress” cases with caution or additional risk mitigation requirements.
