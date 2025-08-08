# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import BytesIO
from fpdf import FPDF
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Banking Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
TICKERS = {
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "J.P. Morgan": "JPM",
    "Citi": "C",
    "Bank of America": "BAC",
    "Barclays (LSE)": "BARC.L"
}

NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY")

# --- CACHING FUNCTIONS ---
@st.cache_data(ttl=600)
def fetch_history(ticker, period="1y", interval="1d"):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    return df.reset_index()

@st.cache_data(ttl=600)
def fetch_info(ticker):
    return yf.Ticker(ticker).info

# --- TECHNICAL INDICATORS ---
def sma(series, window):
    return series.rolling(window).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def beta_volatility(df, benchmark_df):
    returns_stock = df['Close'].pct_change().dropna()
    returns_bench = benchmark_df['Close'].pct_change().dropna()
    beta = np.cov(returns_stock, returns_bench)[0][1] / np.var(returns_bench)
    volatility = returns_stock.std() * np.sqrt(252)
    return beta, volatility

def sharpe_ratio(df, risk_free_rate=0.02):
    returns = df['Close'].pct_change().dropna()
    excess_return = returns - (risk_free_rate / 252)
    return np.sqrt(252) * (excess_return.mean() / excess_return.std())

# --- NEWS FETCH ---
def get_news(query):
    if not NEWS_API_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url, timeout=6).json()
        return res.get("articles", [])
    except:
        return []

# --- SIDEBAR ---
st.sidebar.title("⚙️ Controls")

# Onglets : Dashboard et News
tab = st.sidebar.radio("Onglets", ["Dashboard", "News"])

if tab == "Dashboard":
    mode = st.sidebar.radio("Mode", ["Single Bank", "Comparison"])
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    show_tech = st.sidebar.checkbox("Technical Indicators", value=True)

    st.markdown("<h1 style='color:#004aad;'>📊 Banking Market Intelligence Dashboard</h1>", unsafe_allow_html=True)

    if mode == "Single Bank":
        bank = st.sidebar.selectbox("Select Bank", list(TICKERS.keys()))
        ticker = TICKERS[bank]

        df = fetch_history(ticker, period, interval)
        info = fetch_info(ticker)

        if df.empty or 'Close' not in df.columns or df['Close'].isnull().all():
            st.error("No market data available for this bank and period.")
        else:
            # PRICE CHART WITH CANDLESTICKS
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'
            )])
            fig.update_layout(title=f"{bank} Price Evolution", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # METRICS ROW
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dernier prix", f"{df['Close'].iloc[-1]:.2f} USD")
            c2.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A")
            c3.metric("P/E", info.get("trailingPE", "N/A"))
            dividend_yield = info.get('dividendYield')
            c4.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A")

            # TECHNICAL INDICATORS
            if show_tech:
                df['SMA50'] = sma(df['Close'], 50)
                df['RSI'] = compute_rsi(df['Close'])
                fig2 = px.line(df, x='Date', y=['Close', 'SMA50'], title="Price & SMA50")
                st.plotly_chart(fig2, use_container_width=True)
                st.line_chart(df.set_index('Date')['RSI'].tail(100))

            # RISK METRICS
            benchmark_df = fetch_history("^GSPC", period, interval)
            if benchmark_df.empty or 'Close' not in benchmark_df.columns or benchmark_df['Close'].isnull().all():
                st.warning("Benchmark data unavailable for risk metrics.")
            else:
                beta, vol = beta_volatility(df, benchmark_df)
                st.metric("Beta vs S&P500", f"{beta:.2f}")
                st.metric("Annual Volatility", f"{vol*100:.2f}%")
                st.metric("Sharpe Ratio", f"{sharpe_ratio(df):.2f}")

            # PDF EXPORT
            if st.button("Generate PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(0, 10, f"{bank} - Financial Report", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Last Price: {df['Close'].iloc[-1]:.2f} USD", ln=True)
                pdf.cell(0, 10, f"Market Cap: {info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A", ln=True)
                pdf.cell(0, 10, f"Beta: {beta:.2f}" if 'beta' in locals() else "Beta: N/A", ln=True)
                pdf.cell(0, 10, f"Sharpe Ratio: {sharpe_ratio(df):.2f}", ln=True)
                buf = BytesIO()
                pdf.output(buf)
                buf.seek(0)
                st.download_button("Download PDF", buf, file_name=f"{ticker}_report.pdf")

    else:  # Comparison mode
        banks = st.sidebar.multiselect("Select Banks", list(TICKERS.keys()), default=["Goldman Sachs", "Morgan Stanley"])
        data = {}
        for b in banks:
            data[b] = fetch_history(TICKERS[b], period, interval)

        fig = go.Figure()
        for bank, df in data.items():
            if not df.empty and 'Close' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=bank))
        fig.update_layout(title="Comparative Price Evolution")
        st.plotly_chart(fig, use_container_width=True)

        metrics = []
        for b in banks:
            info = fetch_info(TICKERS[b])
            metrics.append({
                "Bank": b,
                "Price": info.get("currentPrice", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "P/E": info.get("trailingPE", "N/A")
            })
        st.dataframe(pd.DataFrame(metrics))

elif tab == "News":
    st.markdown("<h1 style='color:#004aad;'>📰 Latest Banking News</h1>", unsafe_allow_html=True)
    bank_for_news = st.selectbox("Select Bank for News", list(TICKERS.keys()))
    articles = get_news(bank_for_news)
    if not articles:
        st.info("No news available or NEWSAPI_KEY missing/invalid.")
    else:
        for art in articles:
            st.markdown(f"**[{art.get('title')}]({art.get('url')})** — *{art.get('source', {}).get('name', '')}*")
            st.write(art.get("description", ""))
            st.write("---")
