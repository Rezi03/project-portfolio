import streamlit as st
from utils.ga import inject_ga4

# GA4 Tracking
st.set_page_config(page_title="Finance Projects — IB Portfolio", layout="wide")
inject_ga4()

# Custom CSS
st.markdown("""
<style>
    body { background-color: #f8fafc; }
    .main-container {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        text-align: center;
        font-size: 36px;
        margin-bottom: 10px;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 50px;
    }
    .project-card {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .project-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .project-desc {
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
        flex-grow: 1;
    }
    .btn {
        display: inline-block;
        padding: 10px 20px;
        background-color: #0f172a;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 14px;
        transition: background-color 0.2s ease;
        text-align: center;
    }
    .btn:hover {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Container HTML
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("<h1>Finance Projects</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">A curated selection of professional tools for investment banking and finance analytics.</div>', unsafe_allow_html=True)

# Project grid
projects = [
    ("Banking_Market_Intelligence.py", "Banking Market Intelligence Dashboard", "Aggregated market data, macroeconomic indicators, and sector insights in one interactive dashboard."),
    ("1_M&A_Deals_Dashboard.py", "M&A Deals Dashboard", "Live tracking of global mergers & acquisitions activity with real-time updates."),
    ("2_DCF_Lab.py", "DCF Lab", "Interactive discounted cash flow model with customizable assumptions and instant valuation outputs."),
    ("3_Backtester_Equity_Bonds.py", "Equity/Bond Backtester", "Portfolio backtesting tool with key performance metrics: CAGR, Sharpe ratio, and drawdown."),
]

cols = st.columns(4, gap="large")
for i, (file, title, desc) in enumerate(projects):
    with cols[i]:
        st.markdown('<div class="project-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="project-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="project-desc">{desc}</div>', unsafe_allow_html=True)
        st.link_button("Open Project", f"pages/{file}", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
