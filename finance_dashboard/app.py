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
    res = requests.get(url, timeout=6).json()
    return res.get("articles", [])

# --- SIDEBAR ---
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["Single Bank", "Comparison"])
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
show_tech = st.sidebar.checkbox("Technical Indicators", value=True)
show_news = st.sidebar.checkbox("Bank News", value=True)

# --- MAIN ---
st.markdown("<h1 style='color:#004aad;'>📊 Banking Market Intelligence Dashboard</h1>", unsafe_allow_html=True)

if mode == "Single Bank":
    bank = st.sidebar.selectbox("Select Bank", list(TICKERS.keys()))
    ticker = TICKERS[bank]

    df = fetch_history(ticker, period, interval)
    info = fetch_info(ticker)

    if df.empty:
        st.error("No data available.")
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
        c1.metric("Last Price", f"{df['Close'].iloc[-1]:.2f} USD")
        c2.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,}")
        c3.metric("P/E", info.get("trailingPE", "N/A"))
        c4.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

        # TECHNICAL INDICATORS
        if show_tech:
            df['SMA50'] = sma(df['Close'], 50)
            df['RSI'] = compute_rsi(df['Close'])
            fig2 = px.line(df, x='Date', y=['Close', 'SMA50'], title="Price & SMA50")
            st.plotly_chart(fig2, use_container_width=True)
            st.line_chart(df.set_index('Date')['RSI'].tail(100))

        # RISK METRICS
        benchmark_df = fetch_history("^GSPC", period, interval)
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
            pdf.cell(0, 10, f"Market Cap: {info.get('marketCap', 'N/A'):,}", ln=True)
            pdf.cell(0, 10, f"Beta: {beta:.2f} | Sharpe: {sharpe_ratio(df):.2f}", ln=True)
            buf = BytesIO()
            pdf.output(buf)
            buf.seek(0)
            st.download_button("Download PDF", buf, f"{ticker}_report.pdf")

        # NEWS
        if show_news:
            st.subheader("📰 Latest News")
            for art in get_news(bank):
                st.markdown(f"**[{art.get('title')}]({art.get('url')})** — *{art.get('source', {}).get('name', '')}*")
                st.write(art.get("description", ""))
                st.write("---")

else:
    banks = st.sidebar.multiselect("Select Banks", list(TICKERS.keys()), default=["Goldman Sachs", "Morgan Stanley"])
    data = {}
    for b in banks:
        data[b] = fetch_history(TICKERS[b], period, interval)

    fig = go.Figure()
    for bank, df in data.items():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=bank))
    fig.update_layout(title="Comparative Price Evolution")
    st.plotly_chart(fig, use_container_width=True)

    metrics = []
    for b in banks:
        info = fetch_info(TICKERS[b])
        metrics.append({
            "Bank": b,
            "Price": info.get("currentPrice", None),
            "Market Cap": info.get("marketCap", None),
            "P/E": info.get("trailingPE", None)
        })
    st.dataframe(pd.DataFrame(metrics))
