import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# Load environment variables
RAW_JSON_URL = os.getenv("RAW_JSON_URL", "https://raw.githubusercontent.com/Rezi03/project-portfolio/main/projects/ma-dashboard/deals.json")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

st.set_page_config(page_title="M&A Deals Dashboard", layout="wide")

@st.cache_data(ttl=600)
def load_deals():
    try:
        r = requests.get(RAW_JSON_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading deals: {e}")
        return pd.DataFrame()

# Title
st.title("Global Mergers & Acquisitions Dashboard")

# Load data
df = load_deals()

if df.empty:
    st.warning("No deals available at the moment.")
else:
    st.metric("Number of Deals", len(df))
    st.dataframe(df.head(20))

    # Filter section
    st.sidebar.header("Filters")
    min_value = st.sidebar.date_input("From Date", datetime(2023, 1, 1))
    max_value = st.sidebar.date_input("To Date", datetime.today())

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        filtered = df[(df["date"] >= pd.to_datetime(min_value)) & (df["date"] <= pd.to_datetime(max_value))]
    else:
        filtered = df

    st.subheader("Filtered Deals")
    st.dataframe(filtered)

    # Chart example
    if "date" in filtered.columns:
        deals_per_month = filtered.groupby(filtered["date"].dt.to_period("M")).size()
        st.bar_chart(deals_per_month)