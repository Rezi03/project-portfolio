# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from io import BytesIO
import streamlit as st
from fpdf import FPDF

# --- Page config
st.set_page_config(page_title="Live Banking Market Dashboard", layout="wide", initial_sidebar_state="expanded")

api_key = st.secrets["newsapi"]["api_key"]

# --- Constants
TICKERS = {
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "J.P. Morgan": "JPM",
    "Citi": "C",
    "Bank of America": "BAC",
    "Barclays (LSE)": "BARC.L"
}

# --- Helpers
@st.cache_data(ttl=300)
def fetch_history(ticker, period="1y", interval="1d"):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    df = df.reset_index()
    return df

@st.cache_data(ttl=600)
def fetch_info(ticker):
    tk = yf.Ticker(ticker)
    try:
        return tk.info
    except Exception:
        return {}

def sma(series, window):
    return series.rolling(window).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def momentum_score(df):
    try:
        last = df['Close'].iloc[-1]
        ret_1m = df['Close'].pct_change(21).iloc[-1] * 100
        ret_3m = df['Close'].pct_change(63).iloc[-1] * 100
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        pos_sma = (last / sma50 - 1) * 100
        score = 0.5 * ret_1m + 0.35 * ret_3m + 0.15 * pos_sma
        score = int(max(0, min(100, 50 + score)))
        return score
    except Exception:
        return None

def get_news(query, api_key):
    if not api_key:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    try:
        res = requests.get(url, timeout=6).json()
        return res.get("articles", [])
    except Exception:
        return []

# --- Sidebar (controls)
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Single Bank", "Cross Comparison"])
if mode == "Single Bank":
    bank = st.sidebar.selectbox("Select a bank", list(TICKERS.keys()))
    selected_tickers = [TICKERS[bank]]
else:
    banks = st.sidebar.multiselect("Select banks (2 to 4)", list(TICKERS.keys()),
                                   default=["Goldman Sachs", "Morgan Stanley"])
    selected_tickers = [TICKERS[b] for b in banks]

period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
show_tech = st.sidebar.checkbox("Show technical indicators (SMA / RSI)", value=True)
show_news = st.sidebar.checkbox("Show news (NewsAPI)", value=True)

# NEWS API key from secrets (Streamlit secrets or local .streamlit/secrets.toml)
NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else None

# --- Main layout
st.markdown(
    """
    <style>
    /* Minimal Apple / Banking look */
    .stApp { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    .card { background: #ffffff; border-radius: 14px; box-shadow: 0 6px 18px rgba(15, 15, 15, 0.06); padding: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Live Banking Market Dashboard")
st.write("Interactive market dashboard for major banks. Data source: Yahoo Finance & NewsAPI (news).")

# --- Fetch data for selected tickers
dataframes = {}
infos = {}
for tick in selected_tickers:
    df = fetch_history(tick, period=period, interval=interval)
    if df.empty:
        st.warning(f"No data for {tick}.")
    else:
        dataframes[tick] = df
        infos[tick] = fetch_info(tick)

# --- Mode: Cross Comparison
if mode == "Cross Comparison":
    st.header("Cross comparison")
    if len(dataframes) < 1:
        st.info("Select at least one bank.")
    else:
        fig = go.Figure()
        for t, df in dataframes.items():
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=t))
        fig.update_layout(title="Price comparison", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Comparative table
        rows = []
        for t in selected_tickers:
            info = infos.get(t, {})
            rows.append({
                "Ticker": t,
                "Current Price": info.get("currentPrice", None),
                "Market Cap": info.get("marketCap", None),
                "Trailing P/E": info.get("trailingPE", None)
            })
        if rows:
            st.subheader("Quick metrics")
            st.table(pd.DataFrame(rows).set_index("Ticker"))

# --- Mode: Single Bank
else:
    st.header(bank)
    tick = selected_tickers[0]
    df = dataframes.get(tick)
    info = infos.get(tick, {})

    if df is None or df.empty:
        st.info("No data available. Try another ticker or period.")
    else:
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.update_layout(title=f"{tick} - Price", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics in a clean row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Price", f"{df['Close'].iloc[-1]:.2f}")
        mc = info.get("marketCap", "N/A")
        c2.metric("Market Cap", f"{mc:,}" if mc != "N/A" else "N/A")
        c3.metric("Trailing P/E", info.get("trailingPE", "N/A"))
        c4.metric("Dividend Yield", info.get("dividendYield", "N/A"))

        # Technical indicators
        if show_tech:
            df['SMA50'] = sma(df['Close'], 50)
            df['RSI'] = compute_rsi(df['Close'])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name='SMA50'))
            fig2.update_layout(title="Price & SMA50", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("RSI (last 100 points)")
            st.line_chart(df.set_index('Date')['RSI'].tail(100))

        # Momentum score
        score = momentum_score(df)
        if score is not None:
            st.metric("Momentum score (0-100)", score)

        # Downloads
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"{tick}_history.csv", mime="text/csv")

        if st.button("Generate short PDF summary"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"{tick} - Short summary", ln=True)
            pdf.cell(0, 10, f"Last price: {df['Close'].iloc[-1]:.2f}", ln=True)
            pdf.cell(0, 10, f"Momentum score: {score}", ln=True)
            buf = BytesIO()
            pdf.output(buf)
            buf.seek(0)
            st.download_button("Download PDF", data=buf, file_name=f"{tick}_summary.pdf", mime="application/pdf")

        # News
        if show_news:
            st.subheader("Latest news")
            articles = get_news(bank if bank else tick, NEWS_API_KEY)
            if not articles:
                st.info("No news available or NewsAPI key not configured.")
            else:
                for art in articles:
                    st.markdown(f"**[{art.get('title')}]({art.get('url')})** — *{art.get('source',{}).get('name')}*")
                    st.write(art.get("description"))
                    st.write("---")
