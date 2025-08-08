# app.py
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

NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else None

# --- CACHING FUNCTIONS ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(ticker, period="1y", interval="1d"):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={'Date': 'Date'}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

# --- FINANCIAL METRICS ---
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

def annualized_return(df):
    returns = df['Close'].pct_change().dropna()
    avg_daily = returns.mean()
    return (1 + avg_daily) ** 252 - 1

def cagr(df):
    if df.empty:
        return np.nan
    start = df['Close'].iloc[0]
    end = df['Close'].iloc[-1]
    days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    if days <= 0:
        return np.nan
    years = days / 365.25
    return (end / start) ** (1 / years) - 1

def max_drawdown(df):
    prices = df['Close']
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()

def rolling_volatility(df, window=21):
    returns = df['Close'].pct_change().dropna()
    return returns.rolling(window).std() * np.sqrt(252)

def sharpe_ratio(df, risk_free_rate=0.02):
    returns = df['Close'].pct_change().dropna()
    excess_return = returns - (risk_free_rate / 252)
    return np.sqrt(252) * (excess_return.mean() / (excess_return.std() if excess_return.std() != 0 else np.nan))

def beta_volatility(df, benchmark_df):
    try:
        returns_stock = df['Close'].pct_change().dropna()
        returns_bench = benchmark_df['Close'].pct_change().dropna()
        # align
        df_join = pd.concat([returns_stock, returns_bench], axis=1).dropna()
        if df_join.shape[0] < 2:
            return np.nan, np.nan
        cov = np.cov(df_join.iloc[:,0], df_join.iloc[:,1])
        beta = cov[0,1] / cov[1,1] if cov[1,1] != 0 else np.nan
        volatility = returns_stock.std() * np.sqrt(252)
        return beta, volatility
    except Exception:
        return np.nan, np.nan

def value_at_risk(df, confidence=0.95):
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        return np.nan
    return -np.percentile(returns, (1 - confidence) * 100)

# --- NEWS FETCH ---
def get_news(query):
    if not NEWS_API_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url, timeout=6).json()
        if res.get("status") != "ok":
            return []
        return res.get("articles", [])
    except Exception:
        return []

# --- UI: Sidebar ---
st.sidebar.title("Controls")
tab = st.sidebar.radio("Tabs", ["Dashboard", "News"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, yfinance, Plotly")

# --- MAIN ---
if tab == "Dashboard":
    st.markdown("<h1 style='font-family:-apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto;'>Banking Market Intelligence Dashboard</h1>", unsafe_allow_html=True)
    mode = st.sidebar.radio("Mode", ["Single Bank", "Comparison"])
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    show_tech = st.sidebar.checkbox("Show technical indicators", value=True)
    rf_rate = st.sidebar.number_input("Risk-free rate (annual, e.g. 0.02)", value=0.02, step=0.005, format="%.4f")

    if mode == "Single Bank":
        bank = st.sidebar.selectbox("Select bank", list(TICKERS.keys()))
        ticker = TICKERS[bank]
        df = fetch_history(ticker, period, interval)
        info = fetch_info(ticker)

        if df.empty or 'Close' not in df.columns or df['Close'].isnull().all():
            st.error("No market data available for this bank and period.")
        else:
            # Top summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            last_price = df['Close'].iloc[-1]
            col1.metric("Last Price (USD)", f"{last_price:.2f}")
            market_cap = info.get('marketCap', None)
            col2.metric("Market Cap", f"{market_cap:,}" if market_cap else "N/A")
            col3.metric("CAGR (annual)", f"{cagr(df)*100:.2f}%" if not np.isnan(cagr(df)) else "N/A")
            col4.metric("Max Drawdown", f"{max_drawdown(df)*100:.2f}%" if not np.isnan(max_drawdown(df)) else "N/A")
            col5.metric("Sharpe Ratio", f"{sharpe_ratio(df, risk_free_rate=rf_rate):.2f}")

            # Price candlestick
            fig_price = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'
            )])
            fig_price.update_layout(title=f"{bank} Price Evolution", xaxis_rangeslider_visible=False, template="plotly_white", height=480)
            st.plotly_chart(fig_price, use_container_width=True)

            # Secondary charts row
            left, right = st.columns([2,1])
            with left:
                if show_tech:
                    df['SMA50'] = sma(df['Close'], 50)
                    df['SMA200'] = sma(df['Close'], 200)
                    fig_sma = go.Figure()
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name='SMA50', mode='lines'))
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], name='SMA200', mode='lines'))
                    fig_sma.update_layout(title="Price with SMAs", template="plotly_white", height=350)
                    st.plotly_chart(fig_sma, use_container_width=True)

                    df['RSI'] = compute_rsi(df['Close'])
                    fig_rsi = px.line(df, x='Date', y='RSI', title="RSI (14)", template="plotly_white", height=220)
                    fig_rsi.update_yaxes(range=[0,100])
                    st.plotly_chart(fig_rsi, use_container_width=True)
            with right:
                # Risk & distribution
                var95 = value_at_risk(df, confidence=0.95)
                annual_ret = annualized_return(df)
                beta, vol = beta_volatility(df, fetch_history("^GSPC", period, interval))
                st.metric("Annualized Return", f"{annual_ret*100:.2f}%" if not np.isnan(annual_ret) else "N/A")
                st.metric("Annual Volatility", f"{vol*100:.2f}%" if not np.isnan(vol) else "N/A")
                st.metric("Beta vs S&P500", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
                st.metric("VaR 95%", f"{var95*100:.2f}%" if not np.isnan(var95) else "N/A")

                # Returns histogram
                fig_hist = px.histogram(df.assign(Returns=df['Close'].pct_change()), x='Returns', nbins=50, title="Daily Returns Distribution", template="plotly_white", height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

            # Rolling volatility
            df['RollingVol21'] = rolling_volatility(df, window=21)
            fig_roll = px.line(df, x='Date', y='RollingVol21', title="21-day Rolling Volatility (annualized)", template="plotly_white", height=220)
            st.plotly_chart(fig_roll, use_container_width=True)

            # PDF EXPORT - more structured
            st.markdown("### Generate a structured PDF report")
            generate_col1, generate_col2 = st.columns([3,1])
            with generate_col1:
                st.info("The report includes: cover, key metrics, tables and embedded charts.")
            with generate_col2:
                if st.button("Generate PDF Report", key=f"gen_pdf_{ticker}", help="Create and download a structured PDF report"):
                    # Build figures to embed
                    figs_to_save = [
                        ("price", fig_price),
                        ("sma", fig_sma if 'fig_sma' in locals() else fig_price),
                        ("rsi", fig_rsi if 'fig_rsi' in locals() else fig_hist),
                        ("vol", fig_roll),
                        ("hist", fig_hist)
                    ]
                    tmp_files = []
                    try:
                        for name, fig in figs_to_save:
                            try:
                                img_bytes = fig.to_image(format="png", engine="kaleido")
                            except Exception:
                                # fallback to write_image (older API)
                                img_bytes = fig.write_image(format="png")
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tmp.write(img_bytes)
                            tmp.flush()
                            tmp_files.append(tmp.name)
                            tmp.close()

                        # Build PDF with FPDF
                        pdf = FPDF(unit="pt", format="A4")
                        pdf.set_auto_page_break(auto=True, margin=36)
                        # Cover
                        pdf.add_page()
                        pdf.set_font("Helvetica", style="B", size=20)
                        pdf.cell(0, 40, f"{bank} - Financial Report", ln=True, align="C")
                        pdf.set_font("Helvetica", size=12)
                        pdf.ln(10)
                        pdf.multi_cell(0, 14, f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")
                        pdf.ln(8)
                        pdf.set_font("Helvetica", size=11)
                        pdf.multi_cell(0, 14, f"Period: {period} | Interval: {interval}", align="C")

                        # Metrics page
                        pdf.add_page()
                        pdf.set_font("Helvetica", style="B", size=14)
                        pdf.cell(0, 18, "Key Metrics", ln=True)
                        pdf.set_font("Helvetica", size=10)
                        pdf.ln(6)
                        market_cap_str = f"{market_cap:,}" if market_cap else "N/A"
                        pdf.cell(0, 12, f"Last Price: {last_price:.2f} USD", ln=True)
                        pdf.cell(0, 12, f"Market Cap: {market_cap_str}", ln=True)
                        pdf.cell(0, 12, f"CAGR: {cagr(df)*100:.2f}%" if not np.isnan(cagr(df)) else "CAGR: N/A", ln=True)
                        pdf.cell(0, 12, f"Annualized Return: {annual_ret*100:.2f}%" if not np.isnan(annual_ret) else "Annualized Return: N/A", ln=True)
                        pdf.cell(0, 12, f"Annual Volatility: {vol*100:.2f}%" if not np.isnan(vol) else "Annual Volatility: N/A", ln=True)
                        pdf.cell(0, 12, f"Sharpe Ratio: {sharpe_ratio(df, risk_free_rate=rf_rate):.2f}", ln=True)
                        pdf.cell(0, 12, f"Max Drawdown: {max_drawdown(df)*100:.2f}%" if not np.isnan(max_drawdown(df)) else "Max Drawdown: N/A", ln=True)
                        pdf.cell(0, 12, f"Beta vs S&P500: {beta:.2f}" if not np.isnan(beta) else "Beta vs S&P500: N/A", ln=True)
                        pdf.ln(10)

                        # Insert charts pages
                        for path in tmp_files:
                            pdf.add_page()
                            # Fit image width to page margins (A4 ~ 595pt wide), leave margins
                            page_width = pdf.w - 72
                            try:
                                pdf.image(path, x=36, y=80, w=page_width)
                            except Exception:
                                # If image fails, add small note
                                pdf.set_font("Helvetica", size=10)
                                pdf.cell(0, 12, f"Chart {os.path.basename(path)} could not be embedded.", ln=True)

                        pdf_bytes = pdf.output(dest="S").encode("latin1")
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"{ticker}_financial_report.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{ticker}"
                        )
                    finally:
                        # cleanup temp files
                        for p in tmp_files:
                            try:
                                os.remove(p)
                            except Exception:
                                pass

    else:  # Comparison mode
        banks = st.sidebar.multiselect("Select banks", list(TICKERS.keys()), default=["Goldman Sachs", "Morgan Stanley"])
        if len(banks) < 2:
            st.warning("Select at least two banks for comparison.")
        else:
            data = {}
            for b in banks:
                data[b] = fetch_history(TICKERS[b], period, interval)
            # normalize cumulative returns for visual comparison
            fig = go.Figure()
            for bank, dfb in data.items():
                if not dfb.empty and 'Close' in dfb.columns:
                    dfb_sorted = dfb.sort_values('Date')
                    cum = (1 + dfb_sorted['Close'].pct_change().fillna(0)).cumprod()
                    fig.add_trace(go.Scatter(x=dfb_sorted['Date'], y=cum, mode='lines', name=bank))
            fig.update_layout(title="Normalized Cumulative Returns", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics table
            metrics = []
            for b in banks:
                dfb = data[b]
                if dfb.empty:
                    metrics.append({"Bank": b, "CAGR": "N/A", "Sharpe": "N/A", "Volatility": "N/A"})
                    continue
                metrics.append({
                    "Bank": b,
                    "CAGR": f"{cagr(dfb)*100:.2f}%",
                    "Sharpe": f"{sharpe_ratio(dfb, risk_free_rate=rf_rate):.2f}",
                    "Volatility": f"{(dfb['Close'].pct_change().std()*np.sqrt(252))*100:.2f}%"
                })
            st.dataframe(pd.DataFrame(metrics))

            # Correlation heatmap if aligned
            # build returns DataFrame
            returns_df = pd.DataFrame()
            for b in banks:
                dfb = data[b].set_index('Date').sort_index()
                if 'Close' in dfb.columns:
                    returns_df[b] = dfb['Close'].pct_change()
            if returns_df.dropna(how='all').shape[1] > 1:
                corr = returns_df.corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Return Correlation Matrix", template="plotly_white")
                st.plotly_chart(fig_corr, use_container_width=True)

elif tab == "News":
    st.markdown("<h1 style='font-family:-apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto;'>Latest Banking News</h1>", unsafe_allow_html=True)
    bank_for_news = st.selectbox("Select bank for news", list(TICKERS.keys()))
    if not NEWS_API_KEY:
        st.error("NEWSAPI_KEY missing or invalid. Please add a valid key to Streamlit secrets to fetch news.")
        articles = []
    else:
        articles = get_news(bank_for_news)
        if not articles:
            st.info("No news available for this bank at the moment.")
    for art in articles:
        st.markdown(f"**[{art.get('title')}]({art.get('url')})** — *{art.get('source', {}).get('name', '')}*")
        st.write(art.get("description", ""))
        st.write("---")
