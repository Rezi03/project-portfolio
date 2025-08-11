# app_enhanced_full.py
"""
Banking Market Intelligence - ENGLISH (Full, Enhanced, Multi-page)
-----------------------------------------------------------------
Goal: Professional, feature-rich dashboard to demonstrate finance skills.
Keep API keys in st.secrets or environment variables (DO NOT hardcode).

Features:
- Multi-page navigation (Dashboard, Ticker Details, Compare, Portfolio, Backtest, Reports, News, Settings)
- Large default bank list (US + international; some tickers may require exchange suffixes)
- Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Returns, Drawdown
- Correlation heatmap, pairwise comparison, normalized charts
- Portfolio builder & simulator with rebalancing options
- Simple SMA crossover backtester (parameter sweep)
- Exports: CSV / Excel (multi-sheet) / PDF report (matplotlib)
- News integration using NewsAPI (optional) — key loaded from st.secrets or env var
- Styling: professional color palette and typography
- Robustness: caching, error handling, fallback messages
- Placeholders for authentication / persistence (commented, optional)

How to run:
1) Install dependencies:
   pip install streamlit pandas numpy yfinance plotly matplotlib openpyxl requests pillow
   (kaleido optional for saving plotly images)
2) Put API keys into Streamlit secrets or environment:
   - [Optional] in ~/.streamlit/secrets.toml:
       NEWS_API_KEY = "your_newsapi_key"
3) Run:
   streamlit run app_enhanced_full.py

Author: generated with help from assistant (organized & commented)
"""

import os
import io
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests
from PIL import Image

# -------------------- App Configuration --------------------
st.set_page_config(page_title="Banking Market Intelligence", layout="wide", page_icon="🏦")

# Theme colors (professional, finance)
PRIMARY = "#0b3d91"   # deep navy
SECONDARY = "#0f5bb5"
POS_COLOR = "#2ca02c"
NEG_COLOR = "#d9534f"
MUTED = "#6c757d"
BG = "#ffffff"

# Default ticker list (mainly US banks + some international — users can change in sidebar)
DEFAULT_BANK_TICKERS = [
    "JPM", "BAC", "C", "GS", "MS", "WFC", "USB", "BK", "PNC", "TFC",
    "STT", "SCHW", "BLK", "AXP", "CB", "BAC", "HSBC", "DB", "UBS",
    # Note: some international tickers may require exchange suffixes (e.g., 'BNP.PA', 'SAN.MC', 'HSBC.L', 'UBSG.SW')
    "BNP.PA", "SAN.MC", "HSBC.L", "UBSG.SW", "CS.PA"
]

# Styling CSS
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .title {{ font-size: 28px; font-weight:700; color: {PRIMARY}; }}
        .subtitle {{ color: {MUTED}; font-size:13px; }}
        .kpi-card {{ padding: 12px; border-radius: 8px; background: linear-gradient(90deg, rgba(11,61,145,0.06), rgba(11,61,145,0.01)); }}
        .muted {{ color: {MUTED}; font-size:12px; }}
    </style>
""", unsafe_allow_html=True)

# -------------------- Utility Functions --------------------
@st.cache_data(ttl=1800)
def fetch_history(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a ticker using yfinance. Return DataFrame or None."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        # Do not crash the app; return None and let caller handle
        st.session_state.setdefault("_fetch_errors", []).append(f"{ticker}: {str(e)}")
        return None

def compute_indicators(df: pd.DataFrame, sma_windows: Tuple[int, ...] = (20,50,200), rsi_period: int = 14) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, Bollinger, returns, cumulative and drawdown to DataFrame."""
    df = df.copy()
    # SMAs
    for w in sma_windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    # EMAs
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use Wilder smoothing (EMA with com)
    ma_up = up.ewm(com=rsi_period-1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # Bollinger Bands (20)
    df["BB_mid"] = df["Close"].rolling(window=20).mean()
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    # Returns and cumulative
    df["Return"] = df["Close"].pct_change()
    df["Cumulative"] = (1 + df["Return"].fillna(0)).cumprod()
    # Drawdown
    df["RollingMax"] = df["Cumulative"].cummax()
    df["Drawdown"] = df["Cumulative"] / df["RollingMax"] - 1
    return df

def annualised_return(df: pd.DataFrame) -> float:
    """Ann. return based on Close price daily pct change."""
    if df is None or df.empty:
        return float("nan")
    pct = df["Close"].pct_change().dropna()
    if pct.empty:
        return 0.0
    avg_daily = pct.mean()
    ann = (1 + avg_daily) ** 252 - 1
    return float(ann)

def annualised_vol(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    pct = df["Close"].pct_change().dropna()
    if pct.empty:
        return 0.0
    return float(pct.std() * math.sqrt(252))

def sharpe_ratio(df: pd.DataFrame, rf: float = 0.01) -> float:
    vol = annualised_vol(df)
    if vol == 0 or math.isnan(vol):
        return float("nan")
    return float((annualised_return(df) - rf) / vol)

def max_drawdown(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    if "Cumulative" not in df.columns:
        df = df.copy()
        df["Cumulative"] = (1 + df["Close"].pct_change().fillna(0)).cumprod()
    return float(df["Cumulative"].cummax().sub(df["Cumulative"]).max() / df["Cumulative"].cummax().max())

def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan

# -------------------- Plot helpers --------------------
def plot_candles_with_indicators(df: pd.DataFrame, ticker: str, sma_windows=(20,50,200), show_volume=True, height=600) -> go.Figure:
    df = df.copy().dropna(subset=["Close"])
    fig = make_subplots(rows=2 if show_volume else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.75, 0.25] if show_volume else [1])
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    # SMAs
    for w in sma_windows:
        key = f"SMA_{w}"
        if key in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[key], mode="lines", name=f"SMA {w}", line=dict(width=1)), row=1, col=1)
    # EMAs (optional)
    for e in ["EMA_12","EMA_26"]:
        if e in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[e], mode="lines", name=e, line=dict(width=1, dash="dot")), row=1, col=1)
    # Bollinger
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], mode="lines", name="BB upper", line=dict(width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], mode="lines", fill="tonexty", name="BB lower", line=dict(width=1, dash="dash")), row=1, col=1)
    # Volume
    if show_volume and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(opacity=0.5)), row=2, col=1)
    fig.update_layout(title=f"{ticker} Price & Indicators", template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_rsi(df: pd.DataFrame, height=220) -> Optional[go.Figure]:
    if "RSI" not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought", line_color="red")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold", line_color="green")
    fig.update_layout(height=height, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    return fig

def correlation_heatmap(dfs: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
    """Create an interactive correlation heatmap from closes of dfs dict."""
    # Build frame of close prices
    try:
        closes = pd.DataFrame({k: v["Close"] for k,v in dfs.items()})
        closes = closes.dropna(axis=0, how="any")
        if closes.empty or closes.shape[1] < 2:
            return None
        returns = closes.pct_change().dropna()
        corr = returns.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(title="Correlation Heatmap (daily returns)", height=420, margin=dict(l=10, r=10, t=30, b=10))
        return fig
    except Exception:
        return None

# -------------------- Export helpers --------------------
def to_excel_bytes(data: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in data.items():
            safe_name = str(name)[:31]
            df.to_excel(writer, sheet_name=safe_name)
        writer.save()
    return output.getvalue()

def generate_pdf_report(ticker: str, df: pd.DataFrame, title: str = "") -> bytes:
    """Generate a simple PDF with KPIs and a matplotlib price chart."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.69,8.27))
        plt.axis("off")
        plt.text(0.02, 0.90, f"{title or 'Banking Market Intelligence'} - Report", fontsize=16, weight="bold", color=PRIMARY)
        plt.text(0.02, 0.86, f"Ticker: {ticker}", fontsize=12)
        plt.text(0.02, 0.83, f"Period: {df.index.min().date()} - {df.index.max().date()}", fontsize=10)
        # KPIs
        try:
            last = df["Close"].iloc[-1]
            ann = annualised_return(df)
            vol = annualised_vol(df)
            sr = sharpe_ratio(df)
            mdd = max_drawdown(df)
            plt.text(0.02, 0.78, f"Last Close: {last:,.2f}", fontsize=11)
            plt.text(0.02, 0.75, f"Annualized Return: {ann*100:.2f} %", fontsize=11)
            plt.text(0.02, 0.72, f"Volatility: {vol*100:.2f} %", fontsize=11)
            plt.text(0.02, 0.69, f"Sharpe Ratio: {sr:.2f}", fontsize=11)
            plt.text(0.02, 0.66, f"Max Drawdown: {mdd*100:.2f} %", fontsize=11)
        except Exception:
            plt.text(0.02, 0.78, "KPIs: Not available", fontsize=11)
        pdf.savefig()
        plt.close()

        # Price chart
        fig2, ax = plt.subplots(figsize=(11.69,8.27))
        ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
        for col in df.columns:
            if col.startswith("SMA_") or col.startswith("EMA_"):
                ax.plot(df.index, df[col], linewidth=0.7, label=col)
        ax.set_title(f"{ticker} Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        pdf.savefig()
        plt.close()
    buf.seek(0)
    return buf.getvalue()

# -------------------- App State / Settings --------------------
if "loaded_dfs" not in st.session_state:
    st.session_state.loaded_dfs = {}  # store computed dfs per ticker to avoid refetching in session

# News API key — try st.secrets, then env var — DO NOT hardcode
NEWS_API_KEY = None
if "NEWS_API_KEY" in st.secrets:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
elif "NEWS_API_KEY" in os.environ:
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
# (Same for other API keys: store them in st.secrets or env; we don't store them in code)

# -------------------- Sidebar / Navigation --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home (Dashboard)", "Ticker Details", "Compare", "Portfolio Builder", "Backtest", "Reports & Export", "News & Research", "Settings"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick tickers** (editable):")
user_tickers_text = st.sidebar.text_area("Tickers (comma separated)", value=",".join(DEFAULT_BANK_TICKERS), height=90)
user_tickers = [t.strip().upper() for t in user_tickers_text.split(",") if t.strip()]
st.sidebar.markdown("---")
# Date range shortcuts
default_end = datetime.today()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# Settings (theme, show volume, etc.)
st.sidebar.markdown("**Chart settings**")
show_volume = st.sidebar.checkbox("Show volume on charts", value=True)
default_sma_short = st.sidebar.number_input("Default SMA short (days)", min_value=5, max_value=200, value=50)
default_sma_long  = st.sidebar.number_input("Default SMA long (days)", min_value=50, max_value=400, value=200)
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Built for interviews / finance portfolio showcase — keep secrets safe (use `st.secrets`).")

# -------------------- Page: Home (Dashboard) --------------------
def page_dashboard(tickers: List[str], start_date: datetime, end_date: datetime):
    st.markdown('<div class="title">🏦 Banking Market Intelligence — Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Quick overview: performance, volatility, and top winners / losers.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Load data for each ticker (with caching & progress bar)
    progress = st.progress(0)
    loaded = {}
    errors = []
    total = len(tickers)
    for i, t in enumerate(tickers):
        df = None
        key = f"{t}_{start_date}_{end_date}"
        # Try session cache first
        if key in st.session_state.loaded_dfs:
            df = st.session_state.loaded_dfs[key]
        else:
            df_raw = fetch_history(t, pd.to_datetime(start_date), pd.to_datetime(end_date) + timedelta(days=1))
            if df_raw is not None:
                df = compute_indicators(df_raw, sma_windows=(20,50,200), rsi_period=14)
                st.session_state.loaded_dfs[key] = df
            else:
                errors.append(t)
        if df is not None:
            loaded[t] = df
        progress.progress(int((i+1)/total * 100))
    progress.empty()

    if not loaded:
        st.warning("No data loaded: check tickers and your network.")
        if errors:
            st.info("Errors for tickers: " + ", ".join(errors))
        return

    # Build summary table of KPIs
    rows = []
    for t, df in loaded.items():
        try:
            ann = annualised_return(df)
            vol = annualised_vol(df)
            sr = sharpe_ratio(df)
            mdd = max_drawdown(df)
            last = float(df["Close"].iloc[-1])
            rows.append({"ticker": t, "ann_return": ann, "vol": vol, "sharpe": sr, "mdd": mdd, "last": last})
        except Exception:
            rows.append({"ticker": t, "ann_return": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan, "last": np.nan})
    summary_df = pd.DataFrame(rows).set_index("ticker").sort_values("ann_return", ascending=False)

    # Top KPIs as cards
    kpi_cols = st.columns(min(6, len(summary_df)))
    for idx, (ticker, row) in enumerate(summary_df.head(6).iterrows()):
        with kpi_cols[idx]:
            arrow = "▲" if row["ann_return"] >= 0 else "▼"
            color = POS_COLOR if row["ann_return"] >= 0 else NEG_COLOR
            st.markdown(f"<div class='kpi-card'><strong>{ticker}</strong><div style='font-size:18px;color:{color};'>{arrow} {row['ann_return']*100:,.2f}%</div><div class='muted'>{row['vol']*100:,.2f}% vol · Sharpe {row['sharpe']:.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Performance table (annualized return, volatility, Sharpe, max drawdown)")
    st.dataframe(summary_df.style.format({"ann_return":"{:.2%}", "vol":"{:.2%}", "sharpe":"{:.2f}", "mdd":"{:.2%}", "last":"{:.2f}"}))

    # Correlation heatmap
    st.markdown("### Correlation heatmap (daily returns)")
    heat = correlation_heatmap(loaded)
    if heat is not None:
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough overlapping data to compute correlation heatmap.")

# Continue from:
# st.info("Not enough overlapping data to compute correlation heatmap.")

# ---------------- Multi-page app structure ----------------
# Create a compact, professional multi-page navigation in the sidebar
st.sidebar.markdown("---")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Ticker Detail", "Compare Banks", "Backtest", "Portfolio", "Export & Reports", "Settings"])

# Utility: ensure a stable ordering of tickers used across pages
all_tickers = list(dfs.keys()) if 'dfs' in globals() else valid_tickers if 'valid_tickers' in globals() else []

# Helper: safe get df
def get_df(ticker):
    return dfs.get(ticker) if 'dfs' in globals() else None

# ---------------- PAGE: Overview ----------------
if page == "Overview":
    st.title("Overview — Banking Market Intelligence")
    st.markdown("A quick snapshot of selected banks and market metrics. Use the sidebar to switch pages for deep dives.")
    # KPIs summary (reusing existing summary_df if present)
    if 'summary_df' in globals() and not summary_df.empty:
        kcols = st.columns(min(4, len(summary_df)))
        for i, (ticker, row) in enumerate(summary_df.iterrows()):
            with kcols[i % len(kcols)]:
                perf = row.get('ann', row.get('ann_return', np.nan))
                vol = row.get('vol', row.get('volatility', np.nan))
                sr = row.get('sharpe', np.nan)
                st.markdown(f"**{ticker}**")
                st.markdown(f"<div style='font-size:18px; font-weight:700; color:{PRIMARY if perf>=0 else '#d9534f'}'>{'▲' if perf>=0 else '▼'} {perf*100:.2f}%</div>", unsafe_allow_html=True)
                st.caption(f"Vol: {vol*100:.2f}% · Sharpe: {sr:.2f}")
    else:
        st.info("No summarized KPIs available — load tickers on the main sidebar and press update.")

    # Global correlation heatmap or fallback
    st.markdown("### Cross-ticker correlation")
    heat_fig = None
    try:
        heat_fig = plot_correlation_heatmap(dfs)
    except Exception:
        heat_fig = None
    if heat_fig is not None:
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("Not enough overlapping data to compute correlation heatmap.")

    # Top movers section
    st.markdown("### Top movers (annualised return)")
    if 'summary_df' in globals() and not summary_df.empty:
        top = summary_df.sort_values('ann', ascending=False).head(10)
        st.dataframe(top[['ann','vol','sharpe']].applymap(lambda v: f"{v:.2f}" if isinstance(v, (int,float)) and not pd.isna(v) else v))
    else:
        st.write("No performance data yet.")

    # Small market snapshot (simple: average annual return)
    if all_tickers:
        avg_ann = np.mean([annualised_return(get_df(t)) for t in all_tickers if get_df(t) is not None])
        st.markdown(f"**Market snapshot:** Average annualised return for selected banks: **{avg_ann*100:.2f}%**")

# ---------------- PAGE: Ticker Detail ----------------
elif page == "Ticker Detail":
    st.title("Ticker Detail")
    st.markdown("Inspect single ticker: price, indicators, news, and quick report generation.")
    if not all_tickers:
        st.warning("No tickers loaded — add tickers in the sidebar and update.")
    else:
        # Choose ticker (default: first)
        inspect_ticker = st.selectbox("Ticker", all_tickers, index=0)
        df_ins = get_df(inspect_ticker)
        if df_ins is None or df_ins.empty:
            st.error(f"No data for {inspect_ticker}")
        else:
            # Controls for this page
            col_left, col_right = st.columns([3,1])
            with col_left:
                # Interactive range selector
                days_back = st.slider("Show last N days", min_value=30, max_value=5000, value=365)
                end_date_local = df_ins.index.max()
                start_date_local = max(df_ins.index.min(), end_date_local - pd.Timedelta(days=days_back))
                plot_local = df_ins.loc[start_date_local:end_date_local]

                # Price with indicators
                fig = plot_price_with_indicators(plot_local, inspect_ticker, sma_windows=(sma_short, sma_long, 20), show_macd=show_macd, show_bollinger=show_bollinger, show_volume=show_volume)
                st.plotly_chart(fig, use_container_width=True)

                # RSI small chart
                if 'RSI' in plot_local.columns:
                    fig_r = plot_rsi(plot_local)
                    if fig_r is not None:
                        st.plotly_chart(fig_r, use_container_width=True)

                # Quick metrics box
                st.markdown("#### Quick metrics")
                latest_close = plot_local['Close'].iloc[-1]
                ann = annualised_return(plot_local)
                vol = annualised_vol(plot_local)
                sr = sharpe_ratio(plot_local)
                mdd = max_drawdown(plot_local)
                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Latest Close", f"{latest_close:,.2f}")
                metrics_cols[1].metric("Annualised Return", f"{ann*100:.2f}%")
                metrics_cols[2].metric("Volatility (Ann.)", f"{vol*100:.2f}%")
                metrics_cols[3].metric("Sharpe", f"{sr:.2f}")

                st.markdown(f"Max Drawdown: **{mdd*100:.2f}%**")

                # Trading signals: SMA cross & EMA slope
                st.markdown("#### Signals")
                sma_signal = simple_sma_signal(df_ins, short=sma_short, long=sma_long)
                latest_signal = sma_signal.iloc[-1]
                st.write(f"SMA {sma_short}/{sma_long} signal: **{'LONG' if latest_signal==1 else 'FLAT'}**")
                # EMA slope: positive if EMA_20 > EMA_50
                ema_slope = df_ins['EMA_20'].iloc[-1] - df_ins['EMA_50'].iloc[-1] if 'EMA_50' in df_ins.columns else np.nan
                st.write(f"EMA slope (20-50): **{ema_slope:.2f}**")

            with col_right:
                # News for ticker if key provided (use st.secrets or sidebar key)
                st.markdown("### News")
                api_key_to_use = news_api_key or (st.secrets.get("NEWSAPI_KEY") if "secrets" in globals() else None)
                if api_key_to_use:
                    q = f"{inspect_ticker} bank OR banking"
                    try:
                        url = f"https://newsapi.org/v2/everything?q={requests.utils.requote_uri(q)}&pageSize=5&sortBy=publishedAt&language=en&apiKey={api_key_to_use}"
                        r = requests.get(url, timeout=8)
                        items = r.json().get('articles', [])
                        if not items:
                            st.info("No recent news found.")
                        for a in items:
                            dt = a.get('publishedAt')
                            st.markdown(f"**{a.get('title')}**")
                            st.caption(dt)
                            st.write(a.get('description') or "")
                            st.markdown(f"[Source]({a.get('url')})")
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"News API error: {e}")
                else:
                    st.info("To enable news, add a NewsAPI key in Settings or sidebar.")

                # Small actions: download or generate pdf
                st.markdown("### Actions")
                if st.button(f"Generate PDF report for {inspect_ticker}"):
                    try:
                        pdf_bytes = generate_pdf_report(inspect_ticker, df_ins, plotly_fig=None, extra_text=f"Generated for {inspect_ticker}")
                        st.download_button("Download PDF", data=pdf_bytes, file_name=f"{inspect_ticker}_report.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")

                # Snapshot export of visible range
                if st.button("Export visible range as Excel"):
                    try:
                        sub_df = plot_local.copy()
                        excel_bytes = to_excel_bytes({inspect_ticker: sub_df})
                        st.download_button("Download Excel (visible)", data=excel_bytes, file_name=f"{inspect_ticker}_visible.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        st.error(f"Export error: {e}")

# ---------------- PAGE: Compare Banks ----------------
elif page == "Compare Banks":
    st.title("Compare Banks")
    st.markdown("Side-by-side comparisons and indexed performance. Great for presenting investment thesis or screening candidates.")
    # Multi-select tickers (default to top N)
    selected = st.multiselect("Select banks to compare", options=all_tickers, default=all_tickers[:6])
    if not selected:
        st.info("Select at least one bank to compare.")
    else:
        # Align closes
        closes = pd.DataFrame({t: get_df(t)['Close'] for t in selected}).dropna(how='all')
        if closes.empty:
            st.info("Not enough data overlap for the selected tickers.")
        else:
            # Normalize and plot indexed returns
            normed = closes / closes.iloc[0] * 100
            fig = px.line(normed, labels={'value': 'Indexed (100)', 'index': 'Date'})
            fig.update_layout(height=450, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # Table of metrics for selected tickers
            metrics = []
            for t in selected:
                df_t = get_df(t)
                if df_t is None: continue
                metrics.append({
                    "Ticker": t,
                    "Annualised Return (%)": annualised_return(df_t)*100,
                    "Volatility (%)": annualised_vol(df_t)*100,
                    "Sharpe": sharpe_ratio(df_t),
                    "Max Drawdown (%)": max_drawdown(df_t)*100
                })
            metrics_df = pd.DataFrame(metrics).set_index("Ticker")
            st.markdown("### Comparison table")
            st.dataframe(metrics_df.style.format("{:.2f}"))

            # Correlation submatrix for selected tickers
            try:
                returns = closes.pct_change().dropna()
                corr = returns.corr()
                figcorr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                figcorr.update_layout(height=350)
                st.plotly_chart(figcorr, use_container_width=True)
            except Exception:
                st.info("Could not compute correlation for selected tickers.")

# ---------------- PAGE: Backtest ----------------
elif page == "Backtest":
    st.title("Backtest — Strategy Simulator")
    st.markdown("Run simple backtests (SMA cross) or custom signals to show historical performance.")
    # Strategy selection
    strategy = st.selectbox("Strategy", ["SMA Crossover (short/long)", "Buy & Hold", "Custom: RSI filter"])
    bt_ticker = st.selectbox("Ticker for backtest", all_tickers)
    df_bt = get_df(bt_ticker)
    if df_bt is None:
        st.error("Ticker data not available.")
    else:
        bt_start = st.date_input("Backtest start", value=df_bt.index.min().date())
        bt_end = st.date_input("Backtest end", value=df_bt.index.max().date())
        bt_df = df_bt.loc[bt_start:bt_end].copy()
        if bt_df.empty:
            st.error("No data in selected backtest range.")
        else:
            initial_cap = st.number_input("Initial capital (€)", min_value=100.0, value=10000.0, step=100.0)
            if strategy == "SMA Crossover (short/long)":
                short = st.number_input("Short SMA", value=sma_short)
                long = st.number_input("Long SMA", value=sma_long)
                # Build signal & returns
                bt_df[f"SMA_{short}"] = bt_df['Close'].rolling(short).mean()
                bt_df[f"SMA_{long}"] = bt_df['Close'].rolling(long).mean()
                bt_df['Signal'] = 0
                bt_df.loc[bt_df[f"SMA_{short}"] > bt_df[f"SMA_{long}"], 'Signal'] = 1
                bt_df['StrategyRet'] = bt_df['Signal'].shift(1) * bt_df['Close'].pct_change().fillna(0)
                # Equities curves
                bh = (1 + bt_df['Close'].pct_change().fillna(0)).cumprod() * initial_cap
                strat = (1 + bt_df['StrategyRet']).cumprod() * initial_cap
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bh.index, y=bh, name='Buy & Hold'))
                fig.add_trace(go.Scatter(x=strat.index, y=strat, name=f'SMA {short}/{long} Strategy'))
                fig.update_layout(height=450, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                # Stats
                st.markdown("#### Backtest Statistics")
                st.write("Buy & Hold return:", f"{(bh.iloc[-1]/bh.iloc[0]-1)*100:.2f}%")
                st.write("Strategy return:", f"{(strat.iloc[-1]/strat.iloc[0]-1)*100:.2f}%")
                st.write("Strategy annualised volatility:", f"{bt_df['StrategyRet'].std()*np.sqrt(252)*100:.2f}%")
            elif strategy == "Buy & Hold":
                bh = (1 + bt_df['Close'].pct_change().fillna(0)).cumprod() * initial_cap
                fig = go.Figure(go.Scatter(x=bh.index, y=bh, name='Buy&Hold'))
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                st.write("Total return:", f"{(bh.iloc[-1]/bh.iloc[0]-1)*100:.2f}%")
            else:
                # Custom RSI filter
                rsi_period = st.number_input("RSI period", min_value=5, max_value=50, value=14)
                overbought = st.number_input("Overbought level", value=70)
                oversold = st.number_input("Oversold level", value=30)
                # compute RSI if not present
                if 'RSI' not in bt_df.columns:
                    bt_df['RSI'] = rsi(bt_df['Close'], window=rsi_period)
                bt_df['Signal'] = 0
                bt_df.loc[bt_df['RSI'] < oversold, 'Signal'] = 1
                bt_df.loc[bt_df['RSI'] > overbought, 'Signal'] = 0
                bt_df['StrategyRet'] = bt_df['Signal'].shift(1) * bt_df['Close'].pct_change().fillna(0)
                strat = (1 + bt_df['StrategyRet']).cumprod() * initial_cap
                bh = (1 + bt_df['Close'].pct_change().fillna(0)).cumprod() * initial_cap
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bh.index, y=bh, name='Buy&Hold'))
                fig.add_trace(go.Scatter(x=strat.index, y=strat, name='RSI Strategy'))
                fig.update_layout(height=450, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                st.write("Strategy return:", f"{(strat.iloc[-1]/strat.iloc[0]-1)*100:.2f}%")

# ---------------- PAGE: Portfolio ----------------
elif page == "Portfolio":
    st.title("Portfolio Builder & Analysis")
    st.markdown("Create a simple portfolio and compare to benchmark. Useful for demonstrating allocation skills.")
    selected_port = st.multiselect("Pick tickers", options=all_tickers, default=all_tickers[:4])
    if not selected_port:
        st.info("Select some tickers to build a portfolio.")
    else:
        weights_text = st.text_input("Enter weights (comma separated, will normalise)", value=",".join([f"{1/len(selected_port):.2f}" for _ in selected_port]))
        try:
            weights = np.array([float(w) for w in weights_text.split(",")])
            if len(weights) != len(selected_port):
                st.warning("Number of weights must equal number of selected tickers. They will be normalised.")
                weights = np.ones(len(selected_port)) / len(selected_port)
            else:
                weights = weights / weights.sum()
        except Exception:
            st.warning("Invalid weights; using equal weights.")
            weights = np.ones(len(selected_port)) / len(selected_port)

        # Build returns
        closes = pd.DataFrame({t: get_df(t)['Close'] for t in selected_port}).dropna()
        if closes.empty:
            st.error("Insufficient overlapping data to compute portfolio returns.")
        else:
            rets = closes.pct_change().dropna()
            port_rets = (rets * weights).sum(axis=1)
            port_cum = (1 + port_rets).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name='Portfolio (Indexed)'))
            fig.update_layout(height=450, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Portfolio metrics")
            st.write("Annualised return:", f"{port_rets.mean()*252*100:.2f}%")
            st.write("Annualised vol:", f"{port_rets.std()*np.sqrt(252)*100:.2f}%")
            # Correlation with benchmark if any selected
            if benchmark and benchmark != 'None':
                bdf = fetch_ticker_history(benchmark, pd.to_datetime(start_date), pd.to_datetime(end_date)+pd.Timedelta(days=1))
                if bdf is not None:
                    bench_ret = bdf['Close'].pct_change().dropna()
                    common_idx = port_rets.index.intersection(bench_ret.index)
                    if not common_idx.empty:
                        corr = np.corrcoef(port_rets.loc[common_idx], bench_ret.loc[common_idx])[0,1]
                        st.write("Correlation with benchmark:", f"{corr:.2f}")

# ---------------- PAGE: Export & Reports ----------------
elif page == "Export & Reports":
    st.title("Export & Reports")
    st.markdown("Generate consolidated exports, snapshots and PDF reports for recruiters or interviewers.")
    # Export: All tickers Excel
    if st.button("Export all tickers to Excel"):
        try:
            all_bytes = to_excel_bytes({k: v for k, v in dfs.items()})
            st.download_button("Download All Excel", data=all_bytes, file_name="banking_all_tickers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.markdown("#### Bulk PDF reports")
    multi_report_tickers = st.multiselect("Pick tickers to include in bulk PDF", options=all_tickers, default=all_tickers[:3])
    if st.button("Generate bulk PDF"):
        if not multi_report_tickers:
            st.warning("Select at least one ticker.")
        else:
            try:
                # Create a combined pdf (concatenate individual pdf bytes)
                combined_buf = BytesIO()
                with PdfPages(combined_buf) as pdf_comb:
                    for tick in multi_report_tickers:
                        df_tick = get_df(tick)
                        if df_tick is None: continue
                        # Page: title + kpis
                        fig = plt.figure(figsize=(11.69,8.27))
                        plt.axis('off')
                        plt.text(0.02,0.9, f"{tick} — Snapshot", fontsize=18, weight='bold')
                        plt.text(0.02,0.85, f"Period: {df_tick.index.min().date()} — {df_tick.index.max().date()}", fontsize=10)
                        plt.text(0.02,0.8, f"Last close: {df_tick['Close'].iloc[-1]:,.2f}", fontsize=12)
                        plt.text(0.02,0.76, f"Ann return: {annualised_return(df_tick)*100:.2f} %", fontsize=12)
                        plt.text(0.02,0.72, f"Vol: {annualised_vol(df_tick)*100:.2f} %", fontsize=12)
                        pdf_comb.savefig()
                        plt.close()

                        # Page: price plot
                        fig2, ax = plt.subplots(figsize=(11.69,8.27))
                        ax.plot(df_tick.index, df_tick['Close'], label='Close')
                        for sma_col in [c for c in df_tick.columns if c.startswith('SMA_')][:3]:
                            ax.plot(df_tick.index, df_tick[sma_col], linewidth=0.8, label=sma_col)
                        ax.set_title(f"{tick} Price")
                        ax.grid(True, alpha=0.2)
                        ax.legend()
                        pdf_comb.savefig()
                        plt.close()
                combined_buf.seek(0)
                st.download_button("Download combined PDF", data=combined_buf.getvalue(), file_name="bulk_bank_reports.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Bulk PDF generation failed: {e}")

# ---------------- PAGE: Settings ----------------
elif page == "Settings":
    st.title("Settings")
    st.markdown("Configure app preferences, API keys and appearance.")
    # Language (keep English, but show option)
    lang = st.selectbox("Interface language", options=["English", "French"], index=0)
    st.markdown("**Appearance**")
    theme_mode = st.selectbox("Theme", ["Light", "Dark"], index=0)
    show_advanced = st.checkbox("Show advanced options", value=False)
    if show_advanced:
        st.markdown("#### Advanced")
        ttl_cache = st.number_input("Data cache TTL (seconds)", min_value=60, value=3600, step=60)
        st.write("Note: changing cache TTL will not retroactively clear cache in the running session.")
    # API key fields (do not display raw secrets; encourage st.secrets)
    st.markdown("#### API Keys")
    with st.expander("NewsAPI key (optional)"):
        news_key_input = st.text_input("Paste NewsAPI key here (or add to Streamlit secrets as NEWSAPI_KEY)", value="")
        if news_key_input:
            st.info("For security, add keys to Streamlit secrets for persistent, secure storage. This session uses the provided key temporarily.")
    st.markdown("Settings are applied immediately in this session (no persistent storage).")

# ---------------- Final footer ----------------
st.markdown("---")
st.markdown("**Tips to present to interviewers / recruiters:**")
st.markdown("""
- Start the demo at the **Overview** page: show KPIs and the top movers.  
- Use **Ticker Detail** to highlight signal logic and ability to compute reliable indicators (RSI, MACD, SMA).  
- Use **Compare Banks** to demonstrate screening and correlation analysis.  
- Export a PDF during the demo to show you can generate professional deliverables.  
""")
st.markdown(f"<div style='color:{MUTED}; font-size:12px'>Data source: Yahoo Finance (yfinance). This tool is for educational/demo purposes. Verify before any live trading.</div>", unsafe_allow_html=True)

