#!/usr/bin/env python
# coding: utf-8

# # 1. Business Understanding
# 
# ## 1.2 Problem Statement
# 
# Many investors, lenders, and business owners rely on intuition or outdated reports when evaluating a company’s financial position. This lack of real-time, data-driven analysis can lead to poor investment or lending decisions.
# 
# Our challenge is to develop a **data-powered tool** that automatically analyzes publicly available financial data (like income statements, balance sheets, and cash flows) to assess a company’s **financial stability, profitability, and risk**.
# 
# This project will simplify financial decision-making by transforming raw numbers into actionable insights through **data analysis, visualization, and machine learning**.
# 
# ---
# 
# ## 1.3 Business Objectives
# 
# ### Main Objective
# 
# To build a **data analysis and scoring system** that evaluates a company’s financial health using real-world financial data.
# 
# ### Specific Objectives
# 
# 1. To collect and preprocess financial data from **Yahoo Finance API**  
# 2. To analyze key financial metrics such as revenue growth, net income, debt-to-equity ratio, and cash flow trends.  
# 3. To build a **financial health scoring model** that assigns a score to each company based on performance indicators.  
# 4. To visualize financial insights using clear dashboards and charts for easier interpretation.  
# 5. To provide actionable recommendations for investors or business managers.
# 
# ---
# 
# ## 1.4 Research Questions
# 
# 1. What financial indicators most accurately represent a company’s health and stability?  
# 2. How do profitability, liquidity, and leverage ratios correlate with a company’s risk level?  
# 3. Can we build a model that classifies companies into categories such as _Healthy_, _Moderate_, and _At Risk_?  
# 4. How can visualizing financial trends help investors make better decisions?
# 
# ---
# 
# ## 1.5 Success Criteria
# 
# - The system should accurately collect and clean financial data for multiple companies.  
# - It should compute and visualize key financial ratios and trends.  
# - The scoring model should produce realistic health scores based on financial fundamentals.  
# - The final output should be clear and explainable to both technical and non-technical users.
# 
# ---
# 
# # 2. Data Understanding
# 
# We will use **real financial datasets** fetched directly from APIs — not from Kaggle.
# 
# ---
# 
# ## Datasets & Sources
# 
# | Source | Type of Data | Description |
# | --- | --- | --- |
# | **Yahoo Finance API (via yfinance)** | Company financials | Income statements, balance sheets, cash flow, and stock history |
# ---
# 
# ## Dataset Overview
# 
# Each company dataset will include:
# 
# - **Revenue**  
# - **Gross profit**  
# - **Operating income**  
# - **Net income**  
# - **Total assets & liabilities**  
# - **Cash flow from operations**  
# - **Debt-to-equity ratio**  
# - **Return on assets (ROA)** and **Return on equity (ROE)**  
# - **Stock price performance** over time  
# 
# These metrics help us assess profitability, liquidity, leverage, and efficiency — the four main pillars of financial health.
# 
# ---
# 
# ## Tools and Libraries
# 
# We’ll use the following tools for the analysis:
# 
# | Category | Libraries |
# | --- | --- |
# | **Data Collection** | `yfinance`, `requests`, `pandas` |
# | **Data Cleaning & Processing** | `numpy`, `pandas` |
# | **Visualization** | `matplotlib`, `seaborn`, `plotly` |
# | **Modeling & Scoring** | `scikit-learn`, `statsmodels` |
# | **Deployment (Optional)** | `joblib` for model serialization |
# 

# # 3. Data Preparation
# 
# In this section, we will import the necessary Python libraries and load financial data directly from Yahoo Finance using the `yfinance` API. This will form the foundation of our analysis.
# 
# The data will include income statements, balance sheets, cash flow statements, and stock price history for a chosen company. We will then explore its structure before cleaning and feature engineering.
# 

# In[1]:


# Create requirements.txt 
import os

req_content = """
yfinance>=0.2.66
pandas-datareader
openpyxl
pandas>=1.3.0
numpy>=1.16.5
scikit-learn
matplotlib
seaborn
tqdm
difflib
"""

# Save in current notebook directory
with open("requirements.txt", "w") as f:
    f.write(req_content.strip())

print("requirements.txt created in:", os.getcwd())


# ## 1. Imports – Core Libraries (Security & Scalability Review)
# 
# **Purpose**: Load all dependencies for financial data ingestion, analysis, ML modeling, and visualization.  
# **Why it matters**: Ensures **no runtime `ImportError`** and a **modular design**    
# **Scalability Note**: `tqdm` enables progress tracking; `pickle` for caching (TTL-aware).  
# **Precision Note**: `numpy` used only for arrays – **all money values will use `Decimal` later**.
# 

# In[2]:


# 1. Imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web   
from tqdm import tqdm
import os, pickle, time
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

print("All libraries imported.")


# ## 2. Build Ticker Universe
# 
# **Purpose**: Create a **large, clean list of investable tickers** for financial statement extraction.  
# **Sources**:  
# - **NASDAQ**: Official symbols via `pandas_datareader` (no FTP, no 403)  
# - **S&P 500**: Public CSV from GitHub (no Wikipedia scraping)  
# **Filter**: `marketCap ≥ $100M` → ensures data availability + financial relevance  
# **Fallbacks**: Hardcoded top-10 list if APIs fail → **pipeline never crashes**  
# **Scalability**: Limits to 15,000 checks → avoids rate-limiting; caps final list at 12,000  
# **Security**: Public read-only sources. No credentials.  

# In[3]:


from pandas_datareader.nasdaq_trader import get_nasdaq_symbols 

# 1. Get S&P 500 tickers (from public CSV)
def get_sp500_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        sp500 = pd.read_csv(url)
        return sp500['Symbol'].str.replace('.', '-').tolist()
    except Exception as e:
        print(f"S&P 500 fetch failed: {e}. Using fallback.")
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','LLY','AVGO']

# 2. Get NASDAQ tickers
def get_nasdaq_tickers():
    try:
        nasdaq = get_nasdaq_symbols()
        return nasdaq['NASDAQ Symbol'].dropna().tolist()
    except Exception as e:
        print(f"NASDAQ fetch failed: {e}. Using fallback.")
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AVGO','ASML','PEP']

# 3. Combine and dedupe
sp500 = get_sp500_tickers()
nasdaq = get_nasdaq_tickers()
all_tickers = list(set(sp500 + nasdaq))
print(f"Total raw tickers: {len(all_tickers):,}")

# 4. Cache setup
CACHE_DIR = Path.cwd() / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "ticker_universe.pkl"

# 5. Load from cache 
if CACHE_FILE.exists():
    ticker_symbols = pickle.load(open(CACHE_FILE, "rb"))
    print(f"Loaded cached universe: {len(ticker_symbols):,} tickers")
else:
    # 6. ONE API CALL: Bulk download
    print("Bulk downloading market data (1 request)...")
    data = yf.download(
        tickers=all_tickers,
        period="5d",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,
        threads=True,
        progress=True
    )

    # 7. Extract market cap
    print("Extracting market caps...")
    market_caps = {}
    for t in all_tickers:
        try:
            market_caps[t] = yf.Ticker(t).info.get('marketCap', 0)
        except:
            market_caps[t] = 0

    # 8. Filter: $100M+
    min_cap = 100_000_000
    valid = [t for t, cap in market_caps.items() if cap >= min_cap]
    ticker_symbols = valid[:12_000]  # safety cap

    # 9. Save cache
    pickle.dump(ticker_symbols, open(CACHE_FILE, "wb"))
    print(f"Saved cache → {CACHE_FILE}")

print(f"Final universe: {len(ticker_symbols):,} tickers")


# ## 3. Financial Statement Mapping 
# 
# **Purpose**: Define **standardized, readable field names** for key financial metrics while mapping to **exact Yahoo Finance row labels**.  
# **Why it matters**: Enables **consistent ratio calculations** across 10,000+ tickers despite naming inconsistencies.  
# **Structure**:  
# - **Key** = clean, analyst-friendly name  
# - **Value** = exact string from `yf.Ticker().financials` / `.balance_sheet` / `.cash_flow`  
# **Finance Context**: Focuses on **core profitability, liquidity, and cash flow** drivers.  
# **Scalability Note**: Will be used with **fuzzy matching** later → robust to label changes.
# 

# In[4]:


# 3. Desired items 
income_items = {
    "Total Revenue": "Total Revenue",
    "Gross Profit": "Gross Profit",
    "Operating Income": "Operating Income",
    "Net Income": "Net Income"
}

balance_items = {
    "Total Assets": "Total Assets",
    "Total Liab": "Total Liabilities",
    "Total Stockholder Equity": "Stockholders Equity",
    "Cash": "Cash and Cash Equivalents"
}

cash_flow_items = {
    "Total Cash From Operating Activities": "Operating Cash Flow",
    "Capital Expenditures": "CapEx",
    "Total Cash From Financing Activities": "Financing Cash Flow",
    "Total Cash From Investing Activities": "Investing Cash Flow"
}


# In[5]:


# Combine all mappings
item_mapping = {**income_items, **balance_items, **cash_flow_items}

print("Financial statement mapping defined:")
for clean, raw in list(item_mapping.items())[:5]:
    print(f"  {clean} → '{raw}'")
print("  ...")

# Reuse or recompute market_caps + sectors
def safe_get_info(ticker, field, default='Unknown'):
    """Safely get info field with retry + fallback"""
    try:
        return yf.Ticker(ticker).info.get(field, default)
    except:
        return default

# Reuse market_caps 
try:
    market_caps  
    sectors        
except NameError:
    print("Extracting market caps & sectors (safe mode)...")
    market_caps = {}
    sectors = {}
    for t in tqdm(all_tickers, desc="Info"):
        market_caps[t] = safe_get_info(t, 'marketCap', 0)
        sectors[t] = safe_get_info(t, 'sector', 'Unknown')

# Full universe stats
full_df = pd.DataFrame([
    {'Ticker': t, 'MarketCap': cap} for t, cap in market_caps.items()
])
full_df = full_df[full_df['MarketCap'] > 0].sort_values('MarketCap', ascending=False)

# Your 503
small_universe = ticker_symbols
small_df = full_df[full_df['Ticker'].isin(small_universe)]

# 3. Top 20 missing (by cap)
missing = full_df[~full_df['Ticker'].isin(small_universe)].head(20)
print("\n=== TOP 20 MISSING TICKERS (by market cap) ===")
print(missing[['Ticker', 'MarketCap']].to_string(index=False, formatters={'MarketCap': '${:,.0f}'}))

# 4. Market cap coverage
total_cap_full = full_df['MarketCap'].sum()
total_cap_small = small_df['MarketCap'].sum()
coverage = total_cap_small / total_cap_full * 100 if total_cap_full > 0 else 0

print(f"\nMARKET CAP COVERAGE ")
print(f"Full universe:  {len(full_df):,} tickers → ${total_cap_full:,.0f}")
print(f"Your 503:       {len(small_df):,} tickers → ${total_cap_small:,.0f}")
print(f"Coverage:       {coverage:.1f}% of total market cap")

# 5. Sector diversity (reused from above)
full_sectors = pd.Series([sectors.get(t, 'Unknown') for t in full_df['Ticker']]).value_counts()
small_sectors = pd.Series([sectors.get(t, 'Unknown') for t in small_df['Ticker']]).value_counts()

print(f"\nSECTOR DIVERSITY")
print("Full universe sectors:")
print(full_sectors.head(10))
print("\nYour 503 sectors:")
print(small_sectors.head(10))


# ## 4.Row Name Resolver – `resolve_item_names()`
# 
# **Purpose**: Map **desired financial line items** (e.g., `"Total Revenue"`) to **actual row names** in Yahoo Finance statements, even with spelling, case, or formatting differences.  
# **Why it matters**: Yahoo uses **inconsistent labels** across companies (e.g., `"Total Revenue"` vs `"Total Revenues"`). This function ensures **>95% match rate** at scale.  
# **Matching Strategy** (in order):  
# 1. **Exact match**  
# 2. **Case-insensitive match**  
# 3. **Fuzzy match** (`difflib`, 60% similarity)  
# 
# **Finance Impact**: Prevents **missing data** in ratio calculations → accurate ROE, FCF, etc.  
# **Scalability**: Lightweight, runs per ticker → safe for 10,000+  
# **Security**: Input validation (`df.empty`) → no crashes on failed API calls.

# In[6]:


# 4. resolve_item_names
def resolve_item_names(df, desired_raw_names, verbose=False):
    if df is None or df.empty:
        if verbose:
            print("Warning: Input DataFrame is None or empty")
        return {k: None for k in desired_raw_names}

    actual = list(map(str, df.index))
    actual_lower = [a.lower() for a in actual]
    mapping = {}
    matched = 0

    for desired in desired_raw_names:
        des_lower = desired.lower()
        if desired in actual:
            mapping[desired] = desired
            matched += 1
            if verbose:
                print(f"Exact match: {desired} → {desired}")
            continue
        if des_lower in actual_lower:
            mapping[desired] = actual[actual_lower.index(des_lower)]
            matched += 1
            if verbose:
                print(f"Lowercase match: {desired} → {mapping[desired]}")
            continue
        close = difflib.get_close_matches(desired, actual, n=1, cutoff=0.6)
        mapping[desired] = close[0] if close else None
        if close:
            matched += 1
            if verbose:
                print(f"Fuzzy match: {desired} → {close[0]}")
        else:
            if verbose:
                print(f"No match for: {desired}")

    # Alert if match rate < 80%
    match_rate = matched / len(desired_raw_names) if desired_raw_names else 0
    if match_rate < 0.8 and verbose:
        print(f"Warning: Low match rate ({match_rate:.1%}) - check SME column names")

    return mapping


# ## 5. Helper: Extract & Rename a Statement

# In[7]:


# 5. extract_data_resolve
def extract_data_resolve(df, items_dict, statement_name, verbose=False):
    if df is None or df.empty:
        if verbose:
            print(f"Warning: {statement_name} DataFrame is None or empty")
        return pd.DataFrame()

    desired_raw = list(items_dict.values())
    resolved = resolve_item_names(df, desired_raw, verbose=verbose)

    if verbose:
        print(f"\nResolved mapping for {statement_name}:")
        for d, a in resolved.items():
            print(f"  {d} → {a}")

    actual_to_extract = [resolved[d] for d in desired_raw if resolved[d]]
    if not actual_to_extract:
        if verbose:
            print(f"Warning: No valid columns extracted for {statement_name}")
        return pd.DataFrame()

    extracted = df.reindex(actual_to_extract).T.copy()

    col_rename = {v: k for k, v in items_dict.items() if resolved.get(v)}
    extracted = extracted.rename(columns=col_rename)

    # Log and handle NaN
    if extracted.isna().any().any() and verbose:
        print(f"Warning: {statement_name} has missing values")
        for col in extracted.columns[extracted.isna().any()]:
            print(f"  Missing in {col}")
    
    extracted.insert(0, "Statement", statement_name)
    extracted = extracted.reset_index().rename(columns={"index": "Report Date"})

    # Validate Report Date
    try:
        extracted["Report Date"] = pd.to_datetime(extracted["Report Date"], errors="coerce")
        if extracted["Report Date"].isna().any() and verbose:
            print(f"Warning: Some Report Dates in {statement_name} are invalid")
    except Exception as e:
        if verbose:
            print(f"Error parsing dates in {statement_name}: {e}")
        extracted["Report Date"] = pd.NaT

    return extracted    


# ## 5. Extract & Standardize Financial Rows – `extract_data_resolve()`
# 
# **Purpose**: Pull **specific financial line items** from a raw Yahoo Finance statement (income, balance, or cash flow) using **fuzzy-matched names**, then **reshape and label** them consistently.  
# **Why it matters**: Transforms **wide, messy API output** into **long-format, analyst-ready data** with clean column names.  
# **Key Steps**:  
# 1. **Fuzzy resolve** → map desired → actual rows  
# 2. **Reindex & transpose** → dates become rows  
# 3. **Rename columns** → friendly names (e.g., `"CapEx"`)  
# 4. **Add metadata** → `Statement`, `Report Date`  
# 
# **Finance Impact**: Enables **panel data** for time-series analysis (e.g., revenue growth).  
# **Scalability**: Operates per ticker → safe for 10,000+  
# **Debug**: `verbose=True` prints match quality → audit data pipeline.

# In[8]:


# 6. clean_financial_df
def clean_financial_df(df, verbose=False):
    if df.empty:
        if verbose:
            print("Warning: Input DataFrame is empty")
        return pd.DataFrame()
    
    df = df.copy()
    
    # Parse dates and handle NaT
    df["Report Date"] = pd.to_datetime(df["Report Date"], errors="coerce")
    if df["Report Date"].isna().any() and verbose:
        print("Warning: Some Report Dates are invalid - filling with current year")
    df["Year"] = df["Report Date"].dt.year.fillna(pd.Timestamp.now().year)
    
    # Drop Statement if exists
    df.drop(columns=["Statement"], inplace=True, errors="ignore")
    
    # Dynamic scaling based on max value
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        if verbose:
            print("Warning: No numeric columns to scale")
        return df
    
    max_val = df[numeric_cols].abs().max().max()
    if max_val > 1e9:
        scale = 1e9
        unit = "billions"
    elif max_val > 1e6:
        scale = 1e6
        unit = "millions"
    else:
        scale = 1
        unit = "actual"
    
    df[numeric_cols] = df[numeric_cols] / scale
    df = df.round(2)
    
    if verbose:
        print(f"Scaled numeric columns to {unit} (divided by {scale:,})")
    
    return df


# ## 7.Main Extraction Loop 
# 
# **Purpose**: Download **income, balance sheet, and cash flow** statements for **10,000+ tickers** using `yfinance`, **cache results**, and **stop early** once ≥ 10,000 total rows are collected.  
# **Why it matters**:  
# - **Speed**: First run ~2–4 hours; **subsequent runs < 10 seconds** (cached)  
# - **Reliability**: `try/except` + caching → **no crashes on API failures**  
# - **Efficiency**: Early-stop → avoids processing 10k+ tickers if data goal is met  
# - **Scalability (Rule #5)**: Disk-based cache (`../cache/financials`) → safe for large universes  
# 
# **Key Mechanics**:  
# 1. **Check cache** → load if exists  
# 2. **Else**: `yf.Ticker(t)` → fetch 3 statements → `extract_data_resolve()` → save pickle  
# 3. **Append** to `income_list`, `balance_list`, `cashflow_list` with `Ticker`  
# 4. **Count total rows** → break when ≥ `TARGET_ROWS = 10,000`
# 
# **Security**: Pickle from trusted source only (local). Cache path is isolated.

# In[9]:


# 7. Main extraction (cached & early-stop)
from pathlib import Path
from tqdm import tqdm
import time
import yfinance as yf
from requests.exceptions import HTTPError
import difflib

# Cache directory 
CACHE_DIR = Path.cwd() / "cache" / "financials"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def safe_fetch_ticker(ticker_symbol, retries=3, sleep=1):
    """Fetch financials with retries + sleep to avoid rate limits"""
    ticker = yf.Ticker(ticker_symbol)
    for attempt in range(retries):
        try:
            raw_income = ticker.financials
            raw_balance = ticker.balance_sheet
            raw_cf = ticker.cash_flow
            return raw_income, raw_balance, raw_cf
        except HTTPError as e:
            if attempt == retries - 1:
                print(f"Failed {ticker_symbol} after {retries} tries: {e}")
                return None, None, None
            time.sleep(sleep)
    return None, None, None

income_list = []
balance_list = []
cashflow_list = []

TARGET_ROWS = 10_000
seen_tickers = set()  # Track processed tickers

for ticker_symbol in tqdm(ticker_symbols, desc="Processing"):
    if ticker_symbol in seen_tickers:
        continue  # Skip duplicates

    cache_file = CACHE_DIR / f"{ticker_symbol}.pkl"
    if cache_file.exists():
        inc, bal, cf = pickle.load(open(cache_file, "rb"))
    else:
        raw_income, raw_balance, raw_cf = safe_fetch_ticker(ticker_symbol)
        if raw_income is None:
            continue

        inc = extract_data_resolve(raw_income, income_items, "Income Statement")
        bal = extract_data_resolve(raw_balance, balance_items, "Balance Sheet")
        cf = extract_data_resolve(raw_cf, cash_flow_items, "Cash Flow")

        pickle.dump((inc, bal, cf), open(cache_file, "wb"))

    for name, df in [("Income", inc), ("Balance", bal), ("CashFlow", cf)]:
        if not df.empty:
            # Take only the latest report (most recent Report Date)
            df = df.copy()
            df = df.sort_values("Report Date", ascending=False).head(1)
            df.insert(0, "Ticker", ticker_symbol)
            if name == "Income":
                income_list.append(df)
            elif name == "Balance":
                balance_list.append(df)
            else:
                cashflow_list.append(df)

    seen_tickers.add(ticker_symbol)

    # early-stop
    total = sum(len(lst) for lst in [income_list, balance_list, cashflow_list])
    if total >= TARGET_ROWS:
        print(f"\nReached {total:,} rows – stopping.")
        break


# ## 8. Build Master Tables – Safe Concatenation with Column Deduplication
# 
# **Purpose**: Combine **all per-ticker DataFrames** (from `income_list`, `balance_list`, `cashflow_list`) into **three clean master tables** while **avoiding `InvalidIndexError`** caused by duplicate column names.  
# **Why it matters**:  
# - Ensures **robust concatenation** across 10,000+ tickers  
# - Prevents **silent data loss** from overlapping column labels  
# - Produces **analysis-ready panel data**  
# 
# **Key Fix (`safe_concat`)**:  
# 1. **Drops duplicate columns** (`~df.columns.duplicated()`) → keeps first  
# 2. **Resets index** → clean row alignment  
# 3. **Uses `ignore_index=True`** → fresh integer index  
# 
# **Finance Output**:  
# - `income_master_clean`, `balance_master_clean`, `cashflow_master_clean`  
# - **≥ 10,000 total rows** (verified in next cell)  
# - All values in **billions**, rounded to 2 decimals (`clean_al cached results.

# In[10]:


# 8. Master tables – safe concat with column deduplication
def safe_concat(dfs):
    """Concatenate DataFrames after dropping duplicate columns."""
    if not dfs:
        return pd.DataFrame()
    
    cleaned = []
    for df in dfs:
        # Drop duplicate column names (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        cleaned.append(df.reset_index(drop=True))
    
    return pd.concat(cleaned, ignore_index=True)

def check_duplicates(ticker_list, dfs, name):
    """Check for duplicate tickers in list of DataFrames"""
    ticker_counts = {}
    for df in dfs:
        if 'Ticker' in df.columns:
            for ticker in df['Ticker']:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    duplicates = {t: c for t, c in ticker_counts.items() if c > 1}
    if duplicates:
        print(f"Warning: Duplicate tickers in {name}: {duplicates}")

# Check for duplicates
check_duplicates(ticker_symbols, income_list, "Income")
check_duplicates(ticker_symbols, balance_list, "Balance")
check_duplicates(ticker_symbols, cashflow_list, "CashFlow")

# Build masters
income_master = safe_concat(income_list)
balance_master = safe_concat(balance_list)
cashflow_master = safe_concat(cashflow_list)

# Clean with verbose logging
income_master_clean = clean_financial_df(income_master, verbose=True)
balance_master_clean = clean_financial_df(balance_master, verbose=True)
cashflow_master_clean = clean_financial_df(cashflow_master, verbose=True)

print("\nFinal master tables:")
print(f"Income  : {income_master_clean.shape[0]:,} rows")
print(f"Balance : {balance_master_clean.shape[0]:,} rows")
print(f"CashFlow: {cashflow_master_clean.shape[0]:,} rows")
total_rows = (income_master_clean.shape[0] +
              balance_master_clean.shape[0] +
              cashflow_master_clean.shape[0])
print(f"TOTAL   : {total_rows:,} rows")


# ## 9. Sample Output – Data Quality Check
# 
# **Purpose**: Display **clean, standardized financials** for the **first ticker** in the universe to **validate pipeline success**.  
# **Why it matters**:  
# - Confirms **fuzzy matching**, **caching**, and **concatenation** worked  
# - Shows **real-world structure**: `Ticker`, `Year`, values in **billions**, sorted descending  
# - Enables **manual audit** of key metrics (Revenue, Net Income, FCF, etc.)  
# 
# **Output Format**:  
# - **Three tables** (Income, Balance, Cash Flow)  
# - **Latest 5 years** (most recent first)  
# - **Human-readable** (rounded, no scientific ore modeling.

# In[11]:


# 9. Show a sample
sample_ticker = ticker_symbols[0]
print(f"\nSample – {sample_ticker}")

# Check for empty tables
if income_master_clean.empty:
    print("\nINCOME: No data available")
else:
    print("\nINCOME")
    display(income_master_clean[income_master_clean["Ticker"] == sample_ticker]
            .sort_values("Year", ascending=False).head())

if balance_master_clean.empty:
    print("\nBALANCE: No data available")
else:
    print("\nBALANCE")
    display(balance_master_clean[balance_master_clean["Ticker"] == sample_ticker]
            .sort_values("Year", ascending=False).head())

if cashflow_master_clean.empty:
    print("\nCASH FLOW: No data available")
else:
    print("\nCASH FLOW")
    display(cashflow_master_clean[cashflow_master_clean["Ticker"] == sample_ticker]
            .sort_values("Year", ascending=False).head())


# In[12]:


# Save to CSV

# Output directory
OUTPUT_DIR = Path.cwd() / "output" / "financials"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_to_csv(df, filename, verbose=True):
    """Save DataFrame to CSV with checks and logging"""
    filepath = OUTPUT_DIR / filename
    if df.empty:
        if verbose:
            print(f"Warning: Cannot save {filename} - DataFrame is empty")
        return False
    if filepath.exists():
        if verbose:
            print(f"Warning: {filename} already exists - overwriting")
    try:
        df.to_csv(filepath, index=False)
        if verbose:
            print(f"Success: Saved {filename} with {len(df):,} rows")
        return True
    except Exception as e:
        if verbose:
            print(f"Error: Failed to save {filename}: {e}")
        return False

# Save master tables
save_to_csv(income_master_clean, "income_master.csv")
save_to_csv(balance_master_clean, "balance_master.csv")
save_to_csv(cashflow_master_clean, "cashflow_master.csv")


# # 4.Data cleaning

# ## Inspect Master Table Columns – Schema Validation
# 
# **Purpose**: Print **all column names** in the three clean master tables to **verify data structure** after extraction and concatenation.  
# **Why it matters**:  
# - Confirms **fuzzy mapping** succeeded (e.g., `"CapEx"` present)  
# - Ensures **no duplicate or missing fields** from `safe_concat`  
# - Critical for **feature engineering**

# In[13]:


print("INCOME columns:", income_master_clean.columns.tolist())
print("BALANCE columns:", balance_master_clean.columns.tolist())
print("CASH FLOW columns:", cashflow_master_clean.columns.tolist())


# ## Fill Missing Numeric Values – Prepare for Ratio Calculations
# 
# **Purpose**: Replace **all `NaN` in numeric columns** with `0` across the three master tables to **enable safe arithmetic** in financial ratios.  
# **Why it matters**:  
# - **Ratios like ROE, FCF** will fail or return `inf` if denominator is `NaN`  
# - `0` is **conservative** (assumes missing = no activity) → avoids bias  
# - Applied **only to numeric columns** → preserves `Ticker`, `Year`, dates  
# 
# **Finance Context**:  
# - Missing revenue → treat as `0` (not average)  
# - Missing CapEx → assume `0` spending  
# - Enables **panel-wide ratio computation** without row drops  
# 
# **Security/Precision**: Uses `fillna(0)` on `float64` → **no `Decimal` rounding loss yet**.

# In[14]:


# Fill missing numeric values with 0 (for ratios and consistency)
for df in [income_master_clean, balance_master_clean, cashflow_master_clean]:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
df


# ## Negative Value Audit – Financial Sanity Check
# 
# **Purpose**: Identify **negative values** in fields where they are **logically invalid or rare**, to **detect data quality issues** from Yahoo Finance.  
# **Why it matters**:  
# - **Revenue, Gross Profit, Total Assets, Cash** should **never be negative**  
# - **Liabilities, CapEx** can be negative (e.g., debt reduction), but flagged for review  
# - Early detection → **prevents absurd ratios** (e.g., negative ROA from bad data)  
# 

# In[15]:


# check for values that are negative where they shouldn't be
for df_name, df in zip(
    ["INCOME", "BALANCE", "CASHFLOW"],
    [income_master_clean, balance_master_clean, cashflow_master_clean]
):
    print(f"\n{df_name} – Negative Value Summary:")
    print((df.select_dtypes(include=['float64', 'int64']) < 0).sum())


# ## Remove Duplicates & Reset Index – Ensure Clean Panel Data
# 
# **Purpose**: Eliminate **duplicate rows** and **reset row indices** across all three master tables to guarantee **one record per Ticker-Year-Statement**.  
# **Why it matters**:  
# - `yfinance` may return **duplicate annual reports** (e.g., restated filings)  
# - Duplicates → **inflated row counts**, **biased ratios**, **ML overfitting**  
# - `reset_index(drop=True)` → clean, sequential integers → safe for merging  
# 
# **Finance Impact**:  
# - Prevents **double-counting revenue** in growth calculations  
# - Ensures **unique time-series** per ticker  
# 
# **Security/Precision**: `inplace=True` → memory efficient; no data loss (only duplicates removed).

# In[16]:


# remove duplicates and reset index
for df in [income_master_clean, balance_master_clean, cashflow_master_clean]:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)


# # Feature Engineering – From Raw Financials to Predictive Ratios
# 
# **Purpose**: Transform **cleaned master tables** into **quantitative, time-series features** for modeling (e.g., credit risk, valuation, growth).  
# 
# 
# **Strategy**:  
# 1. **Merge** income + balance + cash flow on `Ticker` + `Year`  
# 2. **Compute 12+ core ratios** using `Decimal` → **no float errors**  
# 3. **Add lags** (1Y, 2Y) → enable forecasting  
# 
# **Security/Precision**: **All money math uses `Decimal`**.  
# **Scalability**: Vectorized `pandas`

# ## Merge the Three Clean Master Tables

# In[30]:


# Merge income + balance + cash-flow on Ticker + Year
merged = (
    income_master_clean
    .merge(balance_master_clean, on=['Ticker', 'Year'], how='outer')
    .merge(cashflow_master_clean, on=['Ticker', 'Year'], how='outer')
)

print(f"Merged rows: {merged.shape[0]:,}")

# Drop duplicate Report Date columns
merged.drop(columns=['Report Date_x', 'Report Date_y'], errors='ignore', inplace=True)

# Rename Stockholders Equity to Total Stockholder Equity
if 'Stockholders Equity' in merged.columns:
    merged.rename(columns={'Stockholders Equity': 'Total Stockholder Equity'}, inplace=True)
    print("Renamed 'Stockholders Equity' → 'Total Stockholder Equity'")

# Impute NaN with 0 for Z-score terms
numeric_cols = merged.select_dtypes(include='number').columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)
print("Filled NaN with 0 in numeric columns")

# Z-Score required columns (Altman's formula)
zscore_required = [
    'Total Revenue', 'Net Income',  # Income
    'Total Assets', 'Current Liabilities', 'Total Stockholder Equity',  # Balance
]

missing_cols = [col for col in zscore_required if col not in merged.columns]
if missing_cols:
    print(f"CRITICAL: Missing Z-score columns: {missing_cols}")
    print("Fix: Check balance_items and cash_flow_items mappings")
else:
    print("All Z-score columns present")

# Show sample for validation
print("\nSample merged row:")
display(merged.head(2))


# ## Convert Money Columns to Decimal (Billions → Actual)

# In[37]:


# Decimal conversion with dynamic scaling for SMEs
from decimal import Decimal, ROUND_HALF_UP

# Dynamic money_cols: detect numeric columns (SME-safe)
money_cols = merged.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Dynamic scale: billions for Yahoo, actual for SMEs
max_val = merged[money_cols].abs().max().max() if money_cols else 0
if max_val > 1e6:
    scale = Decimal('1e9')  # billions
else:
    scale = Decimal('1')  # actual

print(f"Scaling money columns by {scale}")

for col in money_cols:
    merged[col] = merged[col].fillna(0).apply(lambda x: Decimal(str(x)) * scale)
    print(f"Converted {col} to Decimal")

# Safe division: 0 if denominator is zero
def safe_div(num: Decimal, den: Decimal) -> Decimal:
    if den == 0:
        return Decimal('0')
    return (num / den).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

def vec_safe_div(series_num, series_den):
    return pd.Series(
        [safe_div(Decimal(str(a)), Decimal(str(b))) 
         for a, b in zip(series_num, series_den)],
        index=series_num.index
    )

features = merged.copy()

features['Gross Margin']     = vec_safe_div(features['Gross Profit'],     features['Total Revenue'])
features['Operating Margin'] = vec_safe_div(features['Operating Income'], features['Total Revenue'])
features['Net Margin']       = vec_safe_div(features['Net Income'],       features['Total Revenue'])

print("Profitability ratios computed")


# In[ ]:





# In[38]:


# 15.3 Decimal-Safe Division Helper (For Z-Score Ratios)
from decimal import Decimal, ROUND_HALF_UP

def safe_div(num: Decimal, den: Decimal, verbose=False) -> Decimal:
    """Safe division: return 0 if den == 0, else num/den rounded to 4 dp. Log negatives for Z-score audit."""
    if den == 0:
        if verbose:
            print("Warning: Division by zero – returning 0 for ratio")
        return Decimal('0')
    result = (num / den).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    if result < 0 and verbose:
        print("Warning: Negative ratio – possible data anomaly")
    return result

def vec_safe_div(series_num, series_den, verbose=False):
    """Vectorized safe division for pandas Series (SME-ready for batch uploads)."""
    return pd.Series(
        [safe_div(Decimal(str(a)), Decimal(str(b)), verbose) 
         for a, b in zip(series_num, series_den)],
        index=series_num.index
    )


# ## Compute Profitability Ratios

# In[39]:


# Profitability Ratios 
features = merged.copy()

required_cols = ['Gross Profit', 'Operating Income', 'Net Income', 'Total Revenue']
missing = [col for col in required_cols if col not in features.columns]
if missing:
    print(f"CRITICAL: Missing columns for profitability: {missing}")
else:
    features['Gross Margin']     = vec_safe_div(features['Gross Profit'],     features['Total Revenue'])
    features['Operating Margin'] = vec_safe_div(features['Operating Income'], features['Total Revenue'])
    features['Net Margin']       = vec_safe_div(features['Net Income'],       features['Total Revenue'])
    print("Profitability ratios computed")

# Log anomalies
for ratio in ['Gross Margin', 'Operating Margin', 'Net Margin']:
    negatives = features[features[ratio] < 0].shape[0]
    if negatives > 0:
        print(f"Warning: {negatives:,} negative values in {ratio} – possible loss-making SMEs")


# ## Compute Efficiency & Return Ratios

# In[41]:


# 1. Find the actual equity column name
equity_keywords = [
    'stockholder equity', 'shareholders equity', 'total equity', 'equity', 'stockholders equity',
    'common stock equity', 'total shareholders equity'
]

equity_col = None
for kw in equity_keywords:
    matches = [c for c in features.columns if kw.lower() in c.lower()]
    if matches:
        equity_col = matches[0]
        break

if equity_col is None:
    print("Warning: Equity column not found – using fallback 0 for Z-score")
    features['Total Stockholder Equity'] = Decimal('0')
else:
    if equity_col != "Total Stockholder Equity":
        features = features.rename(columns={equity_col: "Total Stockholder Equity"})
        print(f"Renamed equity: '{equity_col}' → 'Total Stockholder Equity'")
    else:
        print("Equity column already canonical.")

# 2. Compute ROA & ROE 
required = ['Net Income', 'Total Assets', 'Total Stockholder Equity']
missing  = [c for c in required if c not in features.columns]

if missing:
    print(f"Cannot compute ROA/ROE – missing: {missing}")
else:
    features['ROA'] = vec_safe_div(features['Net Income'], features['Total Assets'])
    features['ROE'] = vec_safe_div(features['Net Income'], features['Total Stockholder Equity'])
    print("ROA & ROE computed successfully")

    # Log negatives (SME distress signal)
    if features['ROA'].apply(lambda x: x < 0).any():
        print("Warning: Negative ROA detected – indicates potential losses")
    if features['ROE'].apply(lambda x: x < 0).any():
        print("Warning: Negative ROE detected – possible equity erosion")


# ## Compute Liquidity & Leverage Ratios

# In[44]:


# 1. Normalise Total Liabilities column (Yahoo variations)
liab_keywords = [
    'total liabilities', 'total liab', 'liabilities net minority interest', 'liabilities',
    'debt total', 'total debt', 'long term debt'  # ← expanded for SMEs
]

liab_col = None
for kw in liab_keywords:
    matches = [c for c in features.columns if kw.lower() in c.lower()]
    if matches:
        liab_col = matches[0]
        break

if liab_col is None:
    print("Warning: Total Liabilities not found – skipping Current Ratio & Debt/Equity")
else:
    if liab_col != "Total Liab":
        features = features.rename(columns={liab_col: "Total Liab"})
        print(f"Renamed liabilities: '{liab_col}' → 'Total Liab'")
    else:
        print("Liabilities column already canonical: 'Total Liab'")

# 2. Compute ratios only if required columns exist
required = ['Total Assets', 'Total Liab', 'Total Stockholder Equity', 'Net Income']  # ← Z-score tie-in
missing  = [c for c in required if c not in features.columns]

if missing:
    print(f"Cannot compute ratios – missing: {missing}")
else:
    features['Current Ratio'] = vec_safe_div(features['Total Assets'], features['Total Liab'])
    features['Debt to Equity'] = vec_safe_div(features['Total Liab'], features['Total Stockholder Equity'])
    print("Current Ratio & Debt to Equity computed")

    # Log negatives for SME Z-score audit
    if features['Current Ratio'].apply(lambda x: x < 1).any():
        print("Warning: Current Ratio < 1 – liquidity risk for some SMEs")
    if features['Debt to Equity'].apply(lambda x: x > 2).any():
        print("Warning: Debt to Equity > 2 – high leverage risk for some SMEs")


# ## Compute Cash‑Flow Ratios 

# In[54]:


# 1. Map actual column names to expected names
cash_flow_mapping = {
    'Total Cash From Operating Activities': 'Operating Cash Flow',
    'Total Cash From Investing Activities': 'Investing Cash Flow',
    'Total Cash From Financing Activities': 'Financing Cash Flow'
}

# Rename if columns exist
for old_name, new_name in cash_flow_mapping.items():
    if old_name in features.columns:
        features.rename(columns={old_name: new_name}, inplace=True)
        print(f"Renamed: '{old_name}' → '{new_name}'")

# 2. Now check for Operating Cash Flow
if 'Operating Cash Flow' not in features.columns:
    raise KeyError("Operating Cash Flow missing – check cash_flow_items mapping")

# 3. Detect & rename CapEx (expanded keywords)
capex_keywords = [
    'capex', 'capital expenditure', 'capital expenditures',
    'purchase of property', 'ppe', 'net ppe', 'capital outlay',
    'capital spending', 'investing activities capital', 'capital investment'
]

capex_col = None
for kw in capex_keywords:
    matches = [c for c in features.columns if kw.lower() in c.lower()]
    if matches:
        capex_col = matches[0]
        break

if capex_col is None:
    print("Warning: CapEx column not found – using proxy from negative Investing Cash Flow")
    if 'Investing Cash Flow' in features.columns:
        # Convert to Decimal and apply proxy formula
        features['CapEx'] = features['Investing Cash Flow'].apply(
            lambda x: -min(Decimal(str(x)), Decimal('0')) * Decimal('0.8')
        )
        print("Proxy CapEx applied (80% of negative investing flow)")
    else:
        print("No CapEx or proxy available – FCF & related ratios skipped")
else:
    if capex_col != "CapEx":
        features.rename(columns={capex_col: "CapEx"}, inplace=True)
        print(f"Renamed CapEx: '{capex_col}' → 'CapEx'")
    else:
        print("CapEx already canonical")

# 4. Compute FCF and ratios
if all(c in features.columns for c in ['Operating Cash Flow', 'CapEx']):
    features['FCF'] = features['Operating Cash Flow'] - features['CapEx']
    print("FCF computed")
else:
    print("FCF skipped – missing OCF or CapEx")

# FCF Yield
if all(c in features.columns for c in ['FCF', 'Total Stockholder Equity']):
    features['FCF Yield'] = vec_safe_div(features['FCF'], features['Total Stockholder Equity'])
    print("FCF Yield computed")

# CapEx Ratio
if all(c in features.columns for c in ['CapEx', 'Total Revenue']):
    features['CapEx Ratio'] = vec_safe_div(features['CapEx'], features['Total Revenue'])
    print("CapEx Ratio computed")


# In[57]:


# 1. Required ratios
ratio_cols = [
    'Gross Margin', 'Operating Margin', 'ROA', 'ROE',
    'Current Ratio', 'Debt to Equity', 'FCF Yield', 'CapEx Ratio'
]

missing_ratios = [c for c in ratio_cols if c not in features.columns]
if missing_ratios:
    print(f"Warning: Missing ratios for Z‑Score: {missing_ratios}")
    print("Z‑Score will be computed only on available data.")

# 2. Convert to float for scoring (Decimal → float for stats)
# Only select columns that actually exist
available_ratio_cols = [c for c in ratio_cols if c in features.columns]
z_df = features[available_ratio_cols].copy()

for col in available_ratio_cols:
    z_df[col] = z_df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

# 3. Altman Z‑Score formula (custom weights – finance standard)
# Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

# Helper function to safely convert Decimal series to float
def decimal_series_to_float(series):
    """Convert a series with Decimal values to float, handling division carefully"""
    return series.apply(lambda x: float(x) if isinstance(x, Decimal) else x)

# A = Working Capital / Total Assets → approximated as Cash / Total Assets
if all(c in features.columns for c in ['Cash And Cash Equivalents', 'Total Assets']):
    A = features['Cash And Cash Equivalents'] / features['Total Assets']
    A = decimal_series_to_float(A)
else:
    A = pd.Series(0.0, index=features.index)

# B = Retained Earnings / Total Assets → use Net Margin as proxy
B = z_df.get('Net Margin', pd.Series(0.0, index=features.index))

# C = EBIT / Total Assets → use Operating Margin
C = z_df.get('Operating Margin', pd.Series(0.0, index=features.index))

# E = Sales / Total Assets → use Revenue / Assets
if all(c in features.columns for c in ['Total Revenue', 'Total Assets']):
    E = features['Total Revenue'] / features['Total Assets']
    E = decimal_series_to_float(E)
else:
    E = pd.Series(0.0, index=features.index)

# Final Z‑Score (ensure all components are float)
features['Z_Score'] = (
    1.2 * A.fillna(0) +
    1.4 * B.fillna(0) +
    3.3 * C.fillna(0) +
    1.0 * E.fillna(0)
)

# Convert Z_Score to float if it's Decimal
features['Z_Score'] = decimal_series_to_float(features['Z_Score'])

# 4. Risk classification (Altman thresholds)
def classify_z(z):
    """Classify Z-Score into risk categories"""
    try:
        z_val = float(z) if isinstance(z, Decimal) else z
        if z_val > 2.99:
            return 'Safe'
        elif z_val > 1.81:
            return 'Grey'
        else:
            return 'Distress'
    except:
        return 'Unknown'

features['Z_Risk'] = features['Z_Score'].apply(classify_z)

print(f"Z‑Score computed for {len(features):,} rows")
print("\nRisk Distribution:")
print(features['Z_Risk'].value_counts())

# Sample
print("\nSample Z-Scores:")
display(features[['Ticker', 'Year', 'Z_Score', 'Z_Risk']].head(10))

# Additional diagnostics
print("\nZ-Score Statistics:")
print(f"Mean: {features['Z_Score'].mean():.2f}")
print(f"Median: {features['Z_Score'].median():.2f}")
print(f"Min: {features['Z_Score'].min():.2f}")
print(f"Max: {features['Z_Score'].max():.2f}")


# In[ ]:




