import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import BytesIO
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.cache_data.clear()

st.set_page_config(
    page_title="Banking Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

TICKERS = {
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "J.P. Morgan": "JPM",
    "Citi": "C",
    "Bank of America": "BAC",
    "Barclays (LSE)": "BARC.L"
}
BENCHMARK = "^GSPC"

NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")

def ai_complete(prompt, temperature=0.2, max_tokens=600):
    try:
        if not OPENAI_API_KEY:
            return "ERROR: Missing OPENAI_API_KEY"
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are GPT-5, a senior equity research analyst specialized in global banks."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

def sma(series, window): return series.rolling(window).mean()
def compute_rsi(series, period=14):
    delta = series.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean(); ma_down = down.rolling(period).mean()
    rs = ma_up/ma_down; return 100-(100/(1+rs))
def annualized_return(df):
    r = df['Close'].pct_change().dropna()
    if r.empty: return np.nan
    return (1+r.mean())**252 - 1
def cagr(df):
    if df.empty: return np.nan
    start, end = df['Close'].iloc[0], df['Close'].iloc[-1]
    days = (df['Date'].iloc[-1]-df['Date'].iloc[0]).days
    if days<=0: return np.nan
    return (end/start)**(365.25/days) - 1
def max_drawdown(df):
    prices = df['Close']; roll_max = prices.cummax()
    return ((prices-roll_max)/roll_max).min()
def rolling_volatility(df, window=21):
    r = df['Close'].pct_change().dropna()
    if r.empty: return np.Series(dtype=float)
    return r.rolling(window).std()*np.sqrt(252)
def sharpe_ratio(df, rf=0.02):
    r = df['Close'].pct_change().dropna()
    if r.empty or r.std()==0: return np.nan
    return np.sqrt(252)*( (r-(rf/252)).mean()/ (r-(rf/252)).std() )
def sortino_ratio(df, rf=0.02, period=252):
    r = df['Close'].pct_change().dropna()
    if r.empty: return np.nan
    excess = r-(rf/period)
    neg = r[r<0]
    if neg.empty or neg.std()==0: return np.nan
    return (np.sqrt(period)*excess.mean())/neg.std()
def beta_and_vol(df, bench):
    try:
        rs, rb = df['Close'].pct_change().dropna(), bench['Close'].pct_change().dropna()
        j = pd.concat([rs,rb],axis=1).dropna()
        if j.shape[0]<2: return np.nan,np.nan
        c = j.cov(); b = c.iloc[0,1]/c.iloc[1,1] if c.iloc[1,1]!=0 else np.nan
        v = rs.std()*np.sqrt(252); return b,v
    except: return np.nan,np.nan
def value_at_risk(df, conf=0.95):
    r = df['Close'].pct_change().dropna()
    if r.empty: return np.nan
    return -np.percentile(r,(1-conf)*100)
def alpha_annualized(df, bench, rf=0.02):
    try:
        rs, rb = df['Close'].pct_change().dropna(), bench['Close'].pct_change().dropna()
        j = pd.concat([rs,rb],axis=1).dropna()
        if j.shape[0]<2: return np.nan
        y = j.iloc[:,0]-(rf/252); x = j.iloc[:,1]-(rf/252)
        X = np.vstack([np.ones(len(x)),x]).T
        coef = np.linalg.lstsq(X,y,rcond=None)[0]; return coef[0]*252
    except: return np.nan
def simulate_investment(df):
    r = df['Close'].pct_change().fillna(0)
    return (1+r).cumprod()
def fetch_history(ticker, period="1y", interval="1d"):
    try:
        tk=yf.Ticker(ticker); df=tk.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        return df.reset_index()
    except: return pd.DataFrame()
def fetch_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

st.sidebar.title("Controls")
main_tab = st.sidebar.radio("Tabs", ["Dashboard","AI Health Check"])

if main_tab=="AI Health Check":
    st.subheader("AI Health Check")
    st.write("Secrets loaded:", {
        "HAS_OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": OPENAI_MODEL
    })
    if st.button("Run test"):
        out = ai_complete("Reply only: OK", temperature=0.0, max_tokens=5)
        if out.startswith("ERROR:"):
            st.error(out)
        else:
            st.success(f"AI response: {out}")

elif main_tab=="Dashboard":
    st.subheader("Dashboard Demo")
    bank = st.sidebar.selectbox("Bank", list(TICKERS.keys()))
    df = fetch_history(TICKERS[bank], "6mo","1d")
    bench = fetch_history(BENCHMARK,"6mo","1d")
    if df.empty: st.error("No data")
    else:
        df['RSI']=compute_rsi(df['Close'])
        df['SMA50']=sma(df['Close'],50)
        annual_ret=annualized_return(df)
        beta, vol = beta_and_vol(df,bench)
        sharpe=sharpe_ratio(df)
        st.metric("Annual Return", f"{annual_ret*100:.2f}%")
        st.metric("Beta", f"{beta:.2f}")
        st.metric("Volatility", f"{vol*100:.2f}%")
        st.metric("Sharpe", f"{sharpe:.2f}")
        ai_prompt=f"Bank {bank} | Return {annual_ret*100:.2f}%, Beta {beta:.2f}, Vol {vol*100:.2f}%"
        ai_text=ai_complete(ai_prompt)
        if ai_text.startswith("ERROR:"):
            st.error(ai_text)
        elif ai_text.strip():
            st.markdown(ai_text)
        else:
            st.info("AI unavailable right now.")
