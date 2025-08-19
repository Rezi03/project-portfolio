
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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import json
import zipfile
from random import random

st.cache_data.clear()

st.set_page_config(page_title="Banking Market Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

TICKERS = {
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "J.P. Morgan": "JPM",
    "Citi": "C",
    "Bank of America": "BAC",
    "Wells Fargo": "WFC",
    "Barclays (LSE)": "BARC.L",
    "HSBC": "HSBC",
    "UBS": "UBS",
    "Deutsche Bank": "DB",
    "Santander": "SAN"
}

BENCHMARKS = {
    "S&P 500": "^GSPC",
    "Financials ETF (XLF)": "XLF",
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "Euro Stoxx Banks (SX7E via EXS1.DE proxy)": "EXS1.DE"
}

RATE_CREDIT_PROXIES = ["IEF","SHY","LQD","HYG","XLF"]

def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

NEWS_API_KEY = get_secret("NEWSAPI_KEY", None)
FMP_API_KEY = get_secret("FMP_API_KEY", None)
LINKEDIN_URL = get_secret("LINKEDIN_URL", "")
GITHUB_URL = get_secret("GITHUB_URL", "")
PORTFOLIO_URL = get_secret("PORTFOLIO_URL", "")
EMAIL_URL = get_secret("EMAIL", "")

def format_number(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        if abs(x) >= 1e12:
            return f"{x/1e12:.2f}T"
        if abs(x) >= 1e9:
            return f"{x/1e9:.2f}B"
        if abs(x) >= 1e6:
            return f"{x/1e6:.2f}M"
        if abs(x) >= 1e3:
            return f"{x/1e3:.2f}K"
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)
    except Exception:
        return "N/A"

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_history(ticker, period="3y", interval="1d"):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Date": "Date"})
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_history_multi(tickers, period="3y", interval="1d"):
    try:
        data = yf.download(tickers=" ".join(tickers), period=period, interval=interval, auto_adjust=True, progress=False, group_by="ticker")
        frames = {}
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if t in data.columns.levels[0]:
                    df = data[t].reset_index().rename(columns={"Date":"Date"})
                    frames[t] = df
        else:
            t = tickers[0]
            frames[t] = data.reset_index().rename(columns={"Date":"Date"})
        return frames
    except Exception:
        return {t: pd.DataFrame() for t in tickers}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def returns_from_close(df):
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    r = df["Close"].pct_change().dropna()
    return r

def annualized_return_from_returns(r):
    if r.empty:
        return np.nan
    return (1 + r.mean()) ** 252 - 1

def annualized_vol_from_returns(r):
    if r.empty:
        return np.nan
    return r.std() * np.sqrt(252)

def sharpe_from_returns(r, rf=0.02):
    if r.empty or r.std() == 0:
        return np.nan
    ex = r - rf/252
    return np.sqrt(252) * ex.mean() / ex.std()

def sortino_from_returns(r, rf=0.02):
    if r.empty:
        return np.nan
    ex = r - rf/252
    dr = r.copy()
    dr[dr > 0] = 0
    dstd = dr.std()
    if dstd == 0 or np.isnan(dstd):
        return np.nan
    return np.sqrt(252) * ex.mean() / dstd

def drawdown_series(r):
    if r.empty:
        return pd.Series(dtype=float)
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1
    return dd

def max_drawdown_value(r):
    dd = drawdown_series(r)
    return dd.min() if not dd.empty else np.nan

def value_at_risk_from_returns(r, level=0.95):
    if r.empty:
        return np.nan
    return -np.percentile(r, (1-level)*100)

def expected_shortfall_from_returns(r, level=0.95):
    if r.empty:
        return np.nan
    var = np.percentile(r, (1-level)*100)
    tail = r[r <= var]
    if tail.empty:
        return np.nan
    return -tail.mean()

def beta_alpha_from_returns(asset_r, bench_r, rf=0.02):
    if asset_r.empty or bench_r.empty:
        return np.nan, np.nan
    joined = pd.concat([asset_r, bench_r], axis=1).dropna()
    if joined.shape[0] < 5:
        return np.nan, np.nan
    y = joined.iloc[:,0] - rf/252
    x = joined.iloc[:,1] - rf/252
    X = np.vstack([np.ones(len(x)), x.values]).T
    coef, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
    beta = coef[1]
    alpha_daily = coef[0]
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    return beta, alpha_annual

def rolling_beta(asset_r, bench_r, window=63):
    joined = pd.concat([asset_r, bench_r], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    def cov_beta(x):
        a = x.iloc[:,0]
        b = x.iloc[:,1]
        if b.var() == 0:
            return np.nan
        return a.cov(b)/b.var()
    return joined.rolling(window).apply(cov_beta, raw=False)

def rolling_sharpe(r, window=63, rf=0.02):
    if r.empty:
        return pd.Series(dtype=float)
    def calc(x):
        x = pd.Series(x).dropna()
        if x.std() == 0 or len(x) < 5:
            return np.nan
        ex = x - rf/252
        return math.sqrt(252) * ex.mean() / ex.std()
    return r.rolling(window).apply(calc, raw=True)

def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

def bollinger(series, window=20, n=2):
    m = sma(series, window)
    s = series.rolling(window).std()
    u = m + n*s
    l = m - n*s
    return m, u, l

def rolling_volatility(r, window=21):
    if r.empty:
        return pd.Series(dtype=float)
    return r.rolling(window).std() * np.sqrt(252)

def volatility_cone(r):
    windows = [21, 63, 126, 252]
    rows = []
    for w in windows:
        rv = r.rolling(w).std().dropna() * np.sqrt(252)
        if not rv.empty:
            rows.append({"Window": w, "P10": np.percentile(rv, 10), "P50": np.percentile(rv, 50), "P90": np.percentile(rv, 90), "Current": rv.iloc[-1]})
    return pd.DataFrame(rows)

def monte_carlo_paths(last_price, mu, sigma, days=252, paths=300):
    if np.isnan(last_price) or np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        return pd.DataFrame()
    sims = np.zeros((days, paths))
    for i in range(paths):
        rand = np.random.normal(mu, sigma, days)
        sims[:, i] = last_price * np.cumprod(1+rand)
    idx = pd.RangeIndex(1, days+1)
    return pd.DataFrame(sims, index=idx)

def dca_backtest(price_series, monthly_amount=500.0, weights=None):
    if price_series.empty:
        return pd.DataFrame()
    if weights is None:
        weights = {price_series.columns[0]: 1.0}
    monthly_prices = price_series.resample("M").last().dropna(how="all")
    shares = {t: 0.0 for t in monthly_prices.columns}
    values = []
    for dt, row in monthly_prices.iterrows():
        allocs = {t: monthly_amount*weights.get(t, 0) for t in monthly_prices.columns}
        for t in monthly_prices.columns:
            p = row[t]
            if pd.notna(p) and p > 0 and allocs.get(t, 0) > 0:
                shares[t] += allocs[t] / p
        total_val = 0.0
        for t in monthly_prices.columns:
            p = row[t]
            if pd.notna(p):
                total_val += shares[t] * p
        values.append({"Date": dt, "Value": total_val})
    return pd.DataFrame(values).set_index("Date")

def random_portfolios(returns_df, n=1500, rf=0.02):
    if returns_df.empty or returns_df.shape[1] == 0:
        return pd.DataFrame()
    mean = returns_df.mean()*252
    cov = returns_df.cov()*252
    tickers = list(returns_df.columns)
    res = []
    for _ in range(n):
        w = np.random.rand(len(tickers))
        w = w / w.sum()
        port_ret = float(np.dot(w, mean))
        port_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        sr = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
        res.append({"Return": port_ret, "Volatility": port_vol, "Sharpe": sr, "Weights": json.dumps({t: float(wi) for t, wi in zip(tickers, w)})})
    return pd.DataFrame(res)

def sentiment_score(text):
    if not isinstance(text, str):
        return 0
    pos = ["beats","surge","record","upgrade","outperform","strong","growth","profit","gain","rise","up","bull","acquire","deal","raise"]
    neg = ["miss","lawsuit","down","drop","fall","weak","downgrade","loss","bear","risk","fraud","fine","probe","regulator"]
    t = text.lower()
    s = sum([1 for w in pos if w in t]) - sum([1 for w in neg if w in t])
    return max(-3, min(3, s))

def get_news(query):
    if NEWS_API_KEY:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=12&apiKey={NEWS_API_KEY}"
        try:
            res = requests.get(url, timeout=8).json()
            if res.get("status") == "ok":
                arts = []
                for a in res.get("articles", []):
                    arts.append({"date": a.get("publishedAt","")[:19].replace("T"," "), "title": a.get("title",""), "desc": a.get("description",""), "source": a.get("source",{}).get("name",""), "url": a.get("url","")})
                return arts
        except Exception:
            pass
    try:
        tn = yf.Ticker(query).news
        arts = []
        if isinstance(tn, list):
            for item in tn[:12]:
                title = item.get("title","")
                link = item.get("link","")
                pub = item.get("providerPublishTime", None)
                dt = datetime.utcfromtimestamp(pub).strftime("%Y-%m-%d %H:%M") if isinstance(pub, (int,float)) else ""
                arts.append({"date": dt, "title": title, "desc": "", "source": item.get("publisher",""), "url": link})
        return arts
    except Exception:
        return []

def get_mna_deals(symbol=None, limit=20):
    if not FMP_API_KEY:
        return []
    base = "https://financialmodelingprep.com/api/v4/merger_and_acquisitions"
    params = {"apikey": FMP_API_KEY, "limit": limit}
    if symbol:
        params["symbol"] = symbol
    try:
        res = requests.get(base, params=params, timeout=8)
        if res.status_code != 200:
            return []
        data = res.json()
        rows = []
        for d in data:
            rows.append({
                "date": d.get("date",""),
                "acquirer": d.get("acquirer",""),
                "target": d.get("target",""),
                "value": d.get("value",""),
                "status": d.get("status",""),
                "sector": d.get("sector",""),
                "industry": d.get("industry","")
            })
        return rows
    except Exception:
        return []

def create_placeholder_png(message="Not available"):
    buf = BytesIO()
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def figure_to_png(fig, fallback_msg="Chart not available"):
    try:
        import kaleido
        try:
            return fig.to_image(format="png")
        except Exception:
            return create_placeholder_png(fallback_msg)
    except Exception:
        return create_placeholder_png(fallback_msg)

def export_html_report(title, sections):
    html = ["<html><head><meta charset='utf-8'><title>"+title+"</title></head><body style='font-family:Arial,sans-serif'>"]
    html.append(f"<h1>{title}</h1>")
    html.append(f"<p>Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>")
    for s in sections:
        html.append(f"<h2>{s.get('title','Section')}</h2>")
        if "text" in s and s["text"]:
            html.append(f"<pre>{s['text']}</pre>")
        if "fig" in s and s["fig"] is not None:
            try:
                html.append(s["fig"].to_html(full_html=False, include_plotlyjs='cdn'))
            except Exception:
                pass
        if "table" in s and s["table"] is not None:
            try:
                html.append(pd.DataFrame(s["table"]).to_html(border=0))
            except Exception:
                pass
    html.append("</body></html>")
    return "\n".join(html).encode("utf-8")

def export_pdf_report(filename, cover_title, meta_lines, charts, summary_text, bullets):
    tmp_files = []
    try:
        for name, fig, df, title in charts:
            img_bytes = None
            if fig is not None:
                img_bytes = figure_to_png(fig, fallback_msg=title)
            elif df is not None:
                img_bytes = create_placeholder_png(title)
            else:
                img_bytes = create_placeholder_png(title)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.flush()
            tmp_files.append((tmp.name, title))
            tmp.close()
        pdf = FPDF(unit="pt", format="A4")
        pdf.set_auto_page_break(auto=True, margin=36)
        pdf.add_page()
        pdf.set_font("Helvetica", size=20)
        pdf.cell(0, 36, cover_title, ln=True, align="C")
        pdf.set_font("Helvetica", size=11)
        for m in meta_lines:
            pdf.cell(0, 14, m, ln=True, align="C")
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.cell(0, 18, "Key Summary", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 14, summary_text if summary_text else "")
        pdf.ln(8)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 16, "Highlights", ln=True)
        pdf.set_font("Helvetica", size=10)
        for b in bullets:
            if b:
                pdf.multi_cell(0, 12, "- " + b)
        for path, title in tmp_files:
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 16, title, ln=True)
            page_width = pdf.w - 72
            try:
                pdf.image(path, x=36, y=60, w=page_width)
            except Exception:
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 12, "Unable to embed chart image.")
        pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
        st.download_button(label="Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")
    finally:
        for p, _ in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

def header():
    col1, col2 = st.columns([0.7,0.3])
    with col1:
        st.title("Banking Market Intelligence")
        st.caption("by Rezi Sabashvili")
    with col2:
        st.toggle("Recruiter mode", key="recruiter_mode", value=True)
        st.toggle("Dark mode", key="dark_mode", value=False)
    link_parts = []
    if PORTFOLIO_URL:
        link_parts.append(f"[Portfolio]({PORTFOLIO_URL})")
    if LINKEDIN_URL:
        link_parts.append(f"[LinkedIn]({LINKEDIN_URL})")
    if GITHUB_URL:
        link_parts.append(f"[GitHub]({GITHUB_URL})")
    if EMAIL_URL:
        link_parts.append(f"[Email]({EMAIL_URL})")
    if link_parts:
        st.markdown(" • ".join(link_parts))

def sidebar_controls():
    st.sidebar.title("Controls")
    st.sidebar.caption("Market data via Yahoo Finance")
    mode = st.sidebar.selectbox("Mode", ["Overview", "Technicals", "Risk & Performance", "Portfolio", "Rates & Credit", "News & Sentiment", "M&A Pulse", "Exports", "About"], index=0)
    period = st.sidebar.selectbox("Period", ["3mo","6mo","1y","3y","5y","10y","max"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
    bench_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
    benchmark = BENCHMARKS[bench_name]
    rf_rate = st.sidebar.number_input("Risk-free (annual)", value=0.02, step=0.005, format="%.3f")
    focus = st.sidebar.selectbox("Focus bank", list(TICKERS.keys()), index=0)
    dca_amount = st.sidebar.number_input("DCA per month", value=500.0, step=50.0, format="%.2f")
    mc_toggle = st.sidebar.checkbox("Monte Carlo simulation", value=False)
    return mode, period, interval, benchmark, rf_rate, focus, dca_amount, mc_toggle

def recruiter_quick_takeaways(metrics):
    bullets = []
    ar = metrics.get("ann_ret")
    sr = metrics.get("sharpe")
    mdd = metrics.get("max_dd")
    if not np.isnan(ar) and ar > 0:
        bullets.append("Positive annualized return over the selected window.")
    if not np.isnan(sr) and sr > 1.0:
        bullets.append("Risk-adjusted performance above 1.0 Sharpe.")
    if not np.isnan(mdd) and mdd > -0.30:
        bullets.append("Maximum drawdown contained within 30%.")
    if not bullets:
        bullets.append("Mixed signals; see Risk & Technicals for detail.")
    return bullets

def overview_tab(main_df, bench_df, benchmark, ticker_name, rf_rate):
    if main_df.empty:
        st.error("No data for the selected bank.")
        return None
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    fig = go.Figure()
    if "Open" in main_df.columns and "High" in main_df.columns and "Low" in main_df.columns:
        fig.add_trace(go.Candlestick(x=main_df["Date"], open=main_df["Open"], high=main_df["High"], low=main_df["Low"], close=main_df["Close"], name=ticker_name))
        fig.update_xaxes(rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(x=main_df["Date"], y=main_df["Close"], name=ticker_name, mode="lines"))
    if not bench_df.empty and "Close" in bench_df.columns:
        rel = bench_df.set_index("Date")["Close"]
        rel = rel/rel.dropna().iloc[0] - 1
        fig.add_trace(go.Scatter(x=rel.index, y=rel, name=benchmark+" (Rel)", yaxis="y2", mode="lines"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
    fig.update_layout(template=template, legend=dict(orientation="h"), height=520, margin=dict(l=10,r=10,t=40,b=30))
    st.plotly_chart(fig, use_container_width=True)
    r = returns_from_close(main_df)
    r_b = returns_from_close(bench_df)
    ann_ret = annualized_return_from_returns(r)
    ann_vol = annualized_vol_from_returns(r)
    sharpe = sharpe_from_returns(r, rf=rf_rate)
    sortino = sortino_from_returns(r, rf=rf_rate)
    mdd = max_drawdown_value(r)
    var95 = value_at_risk_from_returns(r, 0.95)
    cvar95 = expected_shortfall_from_returns(r, 0.95)
    beta, alpha = beta_alpha_from_returns(r, r_b, rf=rf_rate)
    last_price = main_df["Close"].dropna().iloc[-1] if "Close" in main_df.columns else np.nan
    info = fetch_info(TICKERS[ticker_name])
    market_cap = info.get("marketCap", None)
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Price", f"{last_price:.2f}")
    k2.metric("Market Cap", format_number(market_cap))
    k3.metric("Ann. Return", f"{ann_ret:.2%}" if not np.isnan(ann_ret) else "N/A")
    k4.metric("Ann. Vol", f"{ann_vol:.2%}" if not np.isnan(ann_vol) else "N/A")
    k5.metric("Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
    k6.metric("Sortino", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")
    k7.metric("Max DD", f"{mdd:.2%}" if not np.isnan(mdd) else "N/A")
    if st.session_state.get("recruiter_mode"):
        st.subheader("Quick takeaways")
        for b in recruiter_quick_takeaways({"ann_ret": ann_ret, "sharpe": sharpe, "max_dd": mdd}):
            st.write("• " + b)
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd, "var95": var95, "cvar95": cvar95, "beta": beta, "alpha": alpha, "price_fig": fig}

def technicals_tab(main_df, ticker_name):
    if main_df.empty:
        st.error("No data for the selected bank.")
        return None, None, None
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    price = main_df["Close"]
    e50 = ema(price, 50)
    e200 = ema(price, 200)
    m, u, l = bollinger(price, 20, 2)
    fig = go.Figure()
    if {"Open","High","Low","Close"}.issubset(set(main_df.columns)):
        fig.add_trace(go.Candlestick(x=main_df["Date"], open=main_df["Open"], high=main_df["High"], low=main_df["Low"], close=main_df["Close"], name="OHLC"))
        fig.update_xaxes(rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(x=main_df["Date"], y=price, name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=main_df["Date"], y=e50, name="EMA 50", mode="lines"))
    fig.add_trace(go.Scatter(x=main_df["Date"], y=e200, name="EMA 200", mode="lines"))
    fig.add_trace(go.Scatter(x=main_df["Date"], y=u, name="Boll Upper", mode="lines"))
    fig.add_trace(go.Scatter(x=main_df["Date"], y=l, name="Boll Lower", mode="lines"))
    fig.update_layout(template=template, legend=dict(orientation="h"), height=480)
    st.plotly_chart(fig, use_container_width=True)
    rsi = compute_rsi(price, 14)
    macd_l, macd_s, macd_h = macd(price)
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", mode="lines"))
    fig_r.add_hline(y=70, line_dash="dot")
    fig_r.add_hline(y=30, line_dash="dot")
    fig_r.update_layout(template=template, height=220, yaxis_title="RSI")
    st.plotly_chart(fig_r, use_container_width=True)
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=macd_l.index, y=macd_l, name="MACD", mode="lines"))
    fig_m.add_trace(go.Scatter(x=macd_s.index, y=macd_s, name="Signal", mode="lines"))
    fig_m.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="Hist"))
    fig_m.update_layout(template=template, legend=dict(orientation="h"), height=260)
    st.plotly_chart(fig_m, use_container_width=True)
    alerts = []
    try:
        if rsi.dropna().iloc[-1] >= 70:
            alerts.append("RSI ≥ 70 suggests potential overbought.")
        if rsi.dropna().iloc[-1] <= 30:
            alerts.append("RSI ≤ 30 suggests potential oversold.")
        if e50.dropna().iloc[-1] > e200.dropna().iloc[-1]:
            alerts.append("EMA 50 above EMA 200 indicates medium-term strength.")
        else:
            alerts.append("EMA 50 below EMA 200 indicates medium-term weakness.")
    except Exception:
        pass
    if alerts:
        st.warning(" | ".join(alerts))
    return fig, fig_r, fig_m

def risk_tab(main_df, bench_df, benchmark, ticker_name, rf_rate, mc_toggle):
    if main_df.empty:
        st.error("No data for the selected bank.")
        return {}
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    r = returns_from_close(main_df)
    r_b = returns_from_close(bench_df)
    ann_ret = annualized_return_from_returns(r)
    ann_vol = annualized_vol_from_returns(r)
    sharpe = sharpe_from_returns(r, rf=rf_rate)
    sortino = sortino_from_returns(r, rf=rf_rate)
    mdd_val = max_drawdown_value(r)
    var95 = value_at_risk_from_returns(r, 0.95)
    es95 = expected_shortfall_from_returns(r, 0.95)
    beta, alpha = beta_alpha_from_returns(r, r_b, rf=rf_rate)
    cone = volatility_cone(r)
    if not cone.empty:
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=cone["Window"], y=cone["P10"], name="P10", mode="lines+markers"))
        figc.add_trace(go.Scatter(x=cone["Window"], y=cone["P50"], name="P50", mode="lines+markers"))
        figc.add_trace(go.Scatter(x=cone["Window"], y=cone["P90"], name="P90", mode="lines+markers"))
        figc.add_trace(go.Scatter(x=cone["Window"], y=cone["Current"], name="Current", mode="lines+markers"))
        figc.update_layout(template=template, xaxis_title="Window (days)", yaxis_title="Ann. Vol")
        st.plotly_chart(figc, use_container_width=True)
    dd = drawdown_series(r)
    if not dd.empty:
        figdd = go.Figure()
        figdd.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown"))
        figdd.update_layout(template=template, yaxis_tickformat=".0%", title="Drawdown Curve", height=260)
        st.plotly_chart(figdd, use_container_width=True)
    if not r.empty and not r_b.empty:
        rb = rolling_beta(r, r_b, 63)
        figrb = go.Figure()
        figrb.add_trace(go.Scatter(x=rb.index, y=rb, mode="lines", name="Rolling Beta (63d)"))
        figrb.update_layout(template=template, title=f"Rolling Beta vs {benchmark}", height=260)
        st.plotly_chart(figrb, use_container_width=True)
    if not r.empty:
        rs = rolling_sharpe(r, 63, rf=rf_rate)
        figrs = go.Figure()
        figrs.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines", name="Rolling Sharpe (63d)"))
        figrs.update_layout(template=template, title="Rolling Sharpe (63d)", height=260)
        st.plotly_chart(figrs, use_container_width=True)
    shocks = [-0.1,-0.2,-0.3]
    last_price = main_df["Close"].dropna().iloc[-1] if "Close" in main_df.columns else np.nan
    rows = []
    for s in shocks:
        rows.append({"Shock": f"{int(s*100)}%", "Price": last_price*(1+s) if not np.isnan(last_price) else np.nan, "Change": s})
    st.subheader("Price stress test")
    st.dataframe(pd.DataFrame(rows).set_index("Shock").style.format({"Price":"{:.2f}","Change":"{:.0%}"}))
    if mc_toggle and not r.empty:
        mu = r.mean()
        sigma = r.std()
        sims = monte_carlo_paths(last_price, mu, sigma, 252, 300)
        if not sims.empty:
            figmc = go.Figure()
            for i in sims.columns[:200]:
                figmc.add_trace(go.Scatter(y=sims[i], mode="lines", line=dict(width=1), showlegend=False))
            figmc.update_layout(template=template, title="Monte Carlo Simulations", height=320)
            st.plotly_chart(figmc, use_container_width=True)
    df_metrics = pd.DataFrame({
        "Ann Return":[ann_ret],
        "Ann Vol":[ann_vol],
        "Sharpe":[sharpe],
        "Sortino":[sortino],
        "Max Drawdown":[mdd_val],
        "VaR 95%":[var95],
        "ES 95%":[es95],
        "Beta":[beta],
        "Alpha (annual)":[alpha]
    }, index=[ticker_name])
    st.dataframe(df_metrics.style.format({"Ann Return":"{:.2%}","Ann Vol":"{:.2%}","Sharpe":"{:.2f}","Sortino":"{:.2f}","Max Drawdown":"{:.2%}","VaR 95%":"{:.2%}","ES 95%":"{:.2%}","Beta":"{:.2f}","Alpha (annual)":"{:.2%}"}))
    return {"metrics_table": df_metrics}

def portfolio_tab(selected_banks, period, interval, benchmark, rf_rate, dca_amount):
    if len(selected_banks) == 0:
        st.info("Select at least one bank.")
        return
    tickers = [TICKERS[b] for b in selected_banks]
    frames = fetch_history_multi(tickers + [benchmark], period, interval)
    prices = {}
    for t in tickers:
        df = frames.get(t, pd.DataFrame())
        if not df.empty and "Close" in df.columns:
            prices[t] = df.set_index("Date")["Close"]
    prices_df = pd.DataFrame(prices).dropna(how="all")
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    if prices_df.empty:
        st.error("No data available for the selected banks.")
        return
    ret = prices_df.pct_change().dropna(how="all")
    colw = {}
    st.subheader("Weights")
    cols = st.columns(min(5, len(tickers)))
    for i, t in enumerate(tickers):
        with cols[i % len(cols)]:
            colw[t] = st.slider(f"{t}", 0.0, 1.0, 1.0/len(tickers))
    sw = sum(colw.values())
    if sw == 0:
        st.error("Sum of weights must be > 0")
        return
    for t in colw:
        colw[t] = colw[t]/sw
    w = pd.Series(colw)
    port_r = (ret[w.index] * w).sum(axis=1).dropna()
    bench_df = frames.get(benchmark, pd.DataFrame())
    bench_r = returns_from_close(bench_df)
    cum_port = (1+port_r).cumprod()
    cum_bench = (1+bench_r.reindex(cum_port.index).fillna(0)).cumprod()
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", mode="lines"))
    figp.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=benchmark, mode="lines"))
    figp.update_layout(template=template, yaxis_title="Growth of 1", legend=dict(orientation="h"))
    st.plotly_chart(figp, use_container_width=True)
    te = np.sqrt(252) * (port_r - bench_r.reindex(port_r.index).fillna(0)).std()
    st.write(f"Tracking Error (annualized): {te:.2%}")
    corr = ret.corr()
    figcorr = px.imshow(corr, text_auto=True, aspect="auto", template=template, color_continuous_scale="RdBu", origin="lower")
    figcorr.update_layout(height=420, title="Return Correlation Matrix")
    st.plotly_chart(figcorr, use_container_width=True)
    dca_df = dca_backtest(prices_df, monthly_amount=dca_amount, weights=w.to_dict())
    if not dca_df.empty:
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=dca_df.index, y=dca_df["Value"], name="DCA Value", mode="lines"))
        figd.update_layout(template=template, yaxis_title="Value", title="DCA Backtest")
        st.plotly_chart(figd, use_container_width=True)
    st.subheader("Efficient frontier (random portfolios)")
    pf = random_portfolios(ret, n=1200, rf=rf_rate)
    if not pf.empty:
        figef = px.scatter(pf, x="Volatility", y="Return", color="Sharpe", template=template, title="Mean-Variance Space", hover_data=["Weights"])
        st.plotly_chart(figef, use_container_width=True)

def rates_credit_tab(period, interval, main_ticker):
    proxies = RATE_CREDIT_PROXIES
    data = fetch_history_multi(proxies, period, interval)
    closes = {}
    for t in proxies:
        df = data.get(t, pd.DataFrame())
        if not df.empty and "Close" in df.columns:
            closes[t] = df.set_index("Date")["Close"]
    dfc = pd.DataFrame(closes).dropna(how="all")
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    if dfc.empty:
        st.info("No rate/credit proxy data.")
    else:
        base = dfc.apply(lambda s: s/s.dropna().iloc[0]-1)
        fig_rates = go.Figure()
        for t in base.columns:
            fig_rates.add_trace(go.Scatter(x=base.index, y=base[t], name=t, mode="lines"))
        fig_rates.update_layout(template=template, yaxis_title="Return since start", legend=dict(orientation="h"))
        st.plotly_chart(fig_rates, use_container_width=True)
    df_main = fetch_history(TICKERS[main_ticker], period, interval)
    if not df_main.empty and not dfc.empty:
        both = pd.concat([returns_from_close(df_main), dfc[["IEF","SHY"]].pct_change()], axis=1).dropna()
        both.columns = ["ASSET","IEF","SHY"]
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(x=both.index, y=(1+both["ASSET"]).cumprod(), name=main_ticker, mode="lines"))
        fig_rel.add_trace(go.Scatter(x=both.index, y=(1+both["IEF"]).cumprod(), name="IEF", mode="lines"))
        fig_rel.add_trace(go.Scatter(x=both.index, y=(1+both["SHY"]).cumprod(), name="SHY", mode="lines"))
        fig_rel.update_layout(template=template, legend=dict(orientation="h"), title="Banks vs Rates proxies")
        st.plotly_chart(fig_rel, use_container_width=True)

def news_sentiment_tab(focus_bank_name):
    sym = TICKERS[focus_bank_name]
    items = get_news(sym)
    if not items:
        st.info("No news available right now.")
        return
    df_news = pd.DataFrame(items)
    df_news["Sentiment"] = df_news["title"].apply(sentiment_score)
    st.dataframe(df_news[["date","title","source","Sentiment"]])
    for _, row in df_news.iterrows():
        with st.expander(row["title"][:120]):
            st.write(row["date"] + " • " + row["source"])
            if row.get("desc"):
                st.write(row["desc"])
            if row.get("url"):
                st.link_button("Open source", row["url"])

def mna_pulse_tab(focus_bank_name):
    if not FMP_API_KEY:
        st.info("Provide FMP_API_KEY in Streamlit secrets to enable M&A Pulse.")
        return
    sym = TICKERS[focus_bank_name]
    deals = get_mna_deals(symbol=sym, limit=20)
    if not deals:
        st.info("No recent M&A items for this symbol.")
        return
    df = pd.DataFrame(deals)
    st.dataframe(df)

def exports_tab(ticker_name, period, interval, benchmark, rf_rate, overview_pack, tech_pack, risk_pack):
    figs = []
    if overview_pack and "price_fig" in overview_pack:
        figs.append(("Price", overview_pack["price_fig"], None, "Price"))
    if tech_pack and tech_pack[0] is not None:
        figs.append(("Technicals", tech_pack[0], None, "Technicals"))
    if tech_pack and tech_pack[1] is not None:
        figs.append(("RSI", tech_pack[1], None, "RSI"))
    if tech_pack and tech_pack[2] is not None:
        figs.append(("MACD", tech_pack[2], None, "MACD"))
    cover = f"{ticker_name} Executive Report"
    meta = [f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", f"Period: {period} | Interval: {interval}", f"Benchmark: {benchmark}", f"Risk-free: {rf_rate:.2%}"]
    lines = []
    if overview_pack:
        if not np.isnan(overview_pack.get("ann_ret", np.nan)):
            lines.append(f"Annualized return: {overview_pack['ann_ret']:.2%}")
        if not np.isnan(overview_pack.get("ann_vol", np.nan)):
            lines.append(f"Annualized volatility: {overview_pack['ann_vol']:.2%}")
        if not np.isnan(overview_pack.get("sharpe", np.nan)):
            lines.append(f"Sharpe: {overview_pack['sharpe']:.2f}")
        if not np.isnan(overview_pack.get("sortino", np.nan)):
            lines.append(f"Sortino: {overview_pack['sortino']:.2f}")
        if not np.isnan(overview_pack.get("mdd", np.nan)):
            lines.append(f"Max drawdown: {overview_pack['mdd']:.2%}")
        if not np.isnan(overview_pack.get("var95", np.nan)):
            lines.append(f"VaR 95%: {overview_pack['var95']:.2%}")
        if not np.isnan(overview_pack.get("cvar95", np.nan)):
            lines.append(f"CVaR 95%: {overview_pack['cvar95']:.2%}")
        if not np.isnan(overview_pack.get("beta", np.nan)):
            lines.append(f"Beta vs bench: {overview_pack['beta']:.2f}")
        if not np.isnan(overview_pack.get("alpha", np.nan)):
            lines.append(f"Alpha (annual): {overview_pack['alpha']:.2%}")
    text = " | ".join(lines) if lines else ""
    if st.button("Export HTML report"):
        content = export_html_report(cover, [{"title":"Summary", "text": text}] + [{"title":t, "fig":f} for _, f, _, t in figs])
        st.download_button("Download HTML", data=content, file_name=f"{TICKERS[ticker_name]}_executive_report.html", mime="text/html")
    st.markdown("")
    if st.button("Export PDF report"):
        export_pdf_report(f"{TICKERS[ticker_name]}_executive_report.pdf", cover, meta, [(n,f,d,t) for n,f,d,t in figs], text, [])

def app():
    header()
    mode, period, interval, benchmark, rf_rate, focus_bank_name, dca_amount, mc_toggle = sidebar_controls()
    main_ticker = TICKERS[focus_bank_name]
    main_df = fetch_history(main_ticker, period, interval)
    bench_df = fetch_history(benchmark, period, interval)
    template = "plotly_dark" if st.session_state.get("dark_mode") else "plotly"
    if mode == "Overview":
        overview_tab(main_df, bench_df, benchmark, focus_bank_name, rf_rate)
    elif mode == "Technicals":
        technicals_tab(main_df, focus_bank_name)
    elif mode == "Risk & Performance":
        risk_tab(main_df, bench_df, benchmark, focus_bank_name, rf_rate, mc_toggle)
    elif mode == "Portfolio":
        banks = st.multiselect("Portfolio banks", list(TICKERS.keys()), default=["Goldman Sachs","Morgan Stanley","J.P. Morgan"])
        portfolio_tab(banks, period, interval, benchmark, rf_rate, dca_amount)
    elif mode == "Rates & Credit":
        rates_credit_tab(period, interval, focus_bank_name)
    elif mode == "News & Sentiment":
        news_sentiment_tab(focus_bank_name)
    elif mode == "M&A Pulse":
        mna_pulse_tab(focus_bank_name)
    elif mode == "Exports":
        pack_ov = overview_tab(main_df, bench_df, benchmark, focus_bank_name, rf_rate)
        pack_te = technicals_tab(main_df, focus_bank_name)
        pack_rk = risk_tab(main_df, bench_df, benchmark, focus_bank_name, rf_rate, mc_toggle)
        exports_tab(focus_bank_name, period, interval, benchmark, rf_rate, pack_ov, pack_te, pack_rk)
    elif mode == "About":
        st.markdown("**Stack**: Streamlit, yfinance, pandas, numpy, plotly, FPDF")
        st.markdown("**Features**: executive KPIs, technical indicators, risk metrics (Sharpe, Sortino, VaR, CVaR, drawdown), rolling analytics, portfolio backtesting and efficient frontier, DCA, correlations, rates and credit proxies, news with sentiment, M&A Pulse (optional API), HTML/PDF export, recruiter mode, dark mode.")
        st.markdown("Links are configurable via Streamlit secrets: LINKEDIN_URL, GITHUB_URL, PORTFOLIO_URL, EMAIL, NEWSAPI_KEY, FMP_API_KEY.")
        st.caption("For demonstration only. Not investment advice.")

app()
