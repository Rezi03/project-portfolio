# app_full_improved.py
# Banking Market Intelligence - version complète et améliorée
# ----------------------------------------------------------
# But : rendre le dashboard "pro" et complet (indicateurs, comparaisons,
#       backtest simple, export Excel/PDF, news, caching, UI soignée)
#
# Usage:
#   pip install streamlit pandas numpy yfinance plotly matplotlib openpyxl requests kaleido
#   streamlit run app_full_improved.py
#
# Notes:
# - La génération PNG depuis Plotly utilise 'kaleido' si disponible. Sinon
#   les graphiques PDF utilisent matplotlib en fallback.
# - Pour activer les news, fournis une clé NewsAPI dans la barre latérale.
# - Le code est conçu pour être robuste et lisible; adapte les styles/brandings.
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests
import base64
import math
import tempfile
import os
from typing import Dict, Tuple

# ---------------- Page config & CSS ----------------
st.set_page_config(page_title="Banking Market Intelligence", layout="wide", page_icon="🏦")

PRIMARY = "#0b3d91"   # deep navy
ACCENT = "#2ca02c"    # muted green
MUTED = "#6c757d"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
}}
header .decoration {{display:none}}
.big-title {{font-size:30px; font-weight:700; color: {PRIMARY};}}
.small-muted {{color: {MUTED}; font-size:12px}}
.kpi {{background: linear-gradient(90deg, rgba(11,61,145,0.08), rgba(11,61,145,0.02)); padding:12px; border-radius:8px}}
.card-num {{font-size:18px; font-weight:700; color: {PRIMARY};}}
</style>
""", unsafe_allow_html=True)

# ---------------- Utilities / Financial functions ----------------
@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch historical data with yfinance. Returns DataFrame or None on error."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération du ticker {ticker}: {e}")
        return None

def compute_indicators(df: pd.DataFrame, sma_windows=(20,50,200), rsi_window=14) -> pd.DataFrame:
    df = df.copy()
    # SMA / EMA
    for w in sma_windows:
        df[f"SMA_{w}"] = df['Close'].rolling(window=w, min_periods=1).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_window-1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_window-1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    bb_m = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = bb_m + 2*bb_std
    df['BB_lower'] = bb_m - 2*bb_std
    # Returns & cumulative
    df['Return'] = df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Return'].fillna(0)).cumprod()
    # Drawdown
    df['RollingMax'] = df['Cumulative'].cummax()
    df['Drawdown'] = df['Cumulative']/df['RollingMax'] - 1
    return df

def annualised_return(df: pd.DataFrame) -> float:
    pct = df['Close'].pct_change().dropna()
    if pct.empty:
        return 0.0
    avg_daily = pct.mean()
    ann = (1 + avg_daily) ** 252 - 1
    return ann

def annualised_vol(df: pd.DataFrame) -> float:
    pct = df['Close'].pct_change().dropna()
    if pct.empty:
        return 0.0
    return pct.std() * np.sqrt(252)

def sharpe_ratio(df: pd.DataFrame, risk_free_rate=0.01) -> float:
    vol = annualised_vol(df)
    if vol == 0:
        return np.nan
    return (annualised_return(df) - risk_free_rate) / vol

def max_drawdown(df: pd.DataFrame) -> float:
    # expects 'Cumulative' column
    if 'Cumulative' not in df.columns:
        df = df.copy()
        df['Cumulative'] = (1 + df['Close'].pct_change().fillna(0)).cumprod()
    rolling_max = df['Cumulative'].cummax()
    drawdown = df['Cumulative']/rolling_max - 1
    return drawdown.min()

def simple_sma_signal(df: pd.DataFrame, short=50, long=200) -> pd.Series:
    """Simple long-only SMA crossover signal (1: long, 0: flat)."""
    s = pd.Series(index=df.index, data=0)
    sma_s = df[f"SMA_{short}"]
    sma_l = df[f"SMA_{long}"]
    s[sma_s > sma_l] = 1
    return s.fillna(0)

# -------------- Plotting helpers --------------
def plot_price_with_indicators(df: pd.DataFrame, ticker: str, sma_windows=(20,50,200), show_macd=True, show_bollinger=True, show_volume=True) -> go.Figure:
    df = df.copy().dropna(subset=['Close'])
    rows = 2 if (show_macd or show_volume) else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25] if rows==2 else [1],
                        vertical_spacing=0.06)
    # Candlestick
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{ticker} Price'),
        row=1, col=1
    )
    # SMAs
    for w in sma_windows:
        if f"SMA_{w}" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{w}"], mode='lines', name=f'SMA{w}', line=dict(width=1)), row=1, col=1)
    # EMA
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA20', line=dict(width=1, dash='dot')), row=1, col=1)
    # Bollinger Bands
    if show_bollinger and 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], fill=None, mode='lines', name='BB_upper', line=dict(width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], fill='tonexty', mode='lines', name='BB_lower', line=dict(width=1, dash='dash')), row=1, col=1)
    # Volume
    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker=dict(opacity=0.5)), row=rows, col=1)
    # MACD
    if show_macd and 'MACD' in df.columns:
        if rows == 1:
            # create a small overlay
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', yaxis='y2'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD_signal'), row=2, col=1)
    fig.update_layout(template='plotly_white', height=650, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_rsi(df: pd.DataFrame) -> go.Figure:
    if 'RSI' not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash='dash', annotation_text='Overbought', line_color='red')
    fig.add_hline(y=30, line_dash='dash', annotation_text='Oversold', line_color='green')
    fig.update_layout(height=220, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    return fig

def plot_correlation_heatmap(dfs: Dict[str, pd.DataFrame]) -> go.Figure:
    closes = pd.DataFrame({k: v['Close'] for k, v in dfs.items()})
    closes = closes.dropna(axis=1, how='all').dropna(axis=0, how='any')
    if closes.empty:
        return None
    returns = closes.pct_change().dropna()
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', labels={'x':'Ticker','y':'Ticker','color':'Corr'})
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ---------- Exports: Excel / CSV / PDF ----------
def to_excel_bytes(dfdict: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in dfdict.items():
            safe_name = name[:31]
            # Avoid writing index as object if it's datetime
            df.to_excel(writer, sheet_name=safe_name)
        writer.save()
    return output.getvalue()

def generate_pdf_report(ticker: str, df: pd.DataFrame, plotly_fig: go.Figure = None, extra_text: str = "") -> bytes:
    """Generates a multi-page PDF including KPIs and a price chart.
       Uses matplotlib to ensure portability."""
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: title + KPIs
        fig = plt.figure(figsize=(11.69,8.27))
        plt.axis('off')
        plt.text(0.02, 0.9, 'Banking Market Intelligence — Rapport', fontsize=18, weight='bold', color=PRIMARY)
        plt.text(0.02, 0.85, f'Ticker: {ticker}', fontsize=12)
        plt.text(0.02, 0.82, f'Période: {df.index.min().date()} — {df.index.max().date()}', fontsize=10, color=MUTED)
        # KPIs
        try:
            last = df['Close'].iloc[-1]
            ann = annualised_return(df)
            vol = annualised_vol(df)
            sr = sharpe_ratio(df)
            mdd = max_drawdown(df)
            plt.text(0.02, 0.75, f'Last Close: {last:,.2f}', fontsize=12)
            plt.text(0.02, 0.72, f'Annualised return: {ann*100:.2f} %', fontsize=12)
            plt.text(0.02, 0.69, f'Volatility: {vol*100:.2f} %', fontsize=12)
            plt.text(0.02, 0.66, f'Sharpe ratio: {sr:.2f}', fontsize=12)
            plt.text(0.02, 0.63, f'Max drawdown: {mdd*100:.2f} %', fontsize=12)
        except Exception:
            plt.text(0.02, 0.75, 'KPIs non disponibles', fontsize=12)
        if extra_text:
            plt.text(0.02, 0.55, extra_text, fontsize=10)
        pdf.savefig()
        plt.close()

        # Page 2: price chart via matplotlib
        fig2, ax = plt.subplots(figsize=(11.69,8.27))
        ax.plot(df.index, df['Close'], label='Close')
        for col in df.columns:
            if col.startswith('SMA_') or col.startswith('EMA_'):
                ax.plot(df.index, df[col], label=col, linewidth=0.8)
        ax.set_title(f'{ticker} Price')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.2)
        ax.legend()
        pdf.savefig()
        plt.close()

    buf.seek(0)
    return buf.getvalue()

# --------------- Sidebar: controls / inputs ----------------
st.sidebar.header("Configuration")
with st.sidebar.form(key='controls'):
    tickers_input = st.text_input('Tickers (séparés par virgule)', value='JPM, BAC, C, GS')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    # date inputs
    end_date = st.date_input('Date de fin', value=datetime.today())
    start_date = st.date_input('Date de début', value=datetime.today() - timedelta(days=365))
    # benchmark
    benchmark = st.selectbox('Benchmark (optionnel)', ['^GSPC', 'XLF', 'None'], index=0)
    # sma windows
    sma_short = st.slider('SMA courte (jours)', min_value=5, max_value=100, value=50)
    sma_long = st.slider('SMA longue (jours)', min_value=50, max_value=400, value=200)
    show_rsi = st.checkbox('Afficher RSI', value=True)
    show_macd = st.checkbox('Afficher MACD', value=True)
    show_bollinger = st.checkbox('Afficher Bollinger', value=True)
    show_volume = st.checkbox('Afficher volume', value=True)
    news_api_key = st.text_input('NewsAPI key (optionnel)')
    generate_backtest = st.checkbox('Activer backtest SMA cross (simple)', value=True)
    submit = st.form_submit_button('Mettre à jour')

if not tickers:
    st.warning("Ajoute au moins un ticker.")
    st.stop()

# --------------- Header & Branding ----------------
col1, col2 = st.columns([5,1])
with col1:
    st.markdown('<div class="big-title">🏦 Banking Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Dashboard — Analyses & rapports — prêt pour entretien / stage</div>', unsafe_allow_html=True)
with col2:
    logo = st.file_uploader('Logo (optionnel)', type=['png','jpg','jpeg'])
    if logo:
        st.image(logo, width=80)

st.markdown('---')

# --------------- Data retrieval & compute ----------------
progress_text = st.empty()
progress_bar = st.progress(0)
dfs = {}
valid_tickers = []
for i, t in enumerate(tickers):
    progress_text.text(f"Chargement: {t} ({i+1}/{len(tickers)})")
    df_raw = fetch_ticker_history(t, pd.to_datetime(start_date), pd.to_datetime(end_date)+pd.Timedelta(days=1))
    if df_raw is None:
        st.warning(f"Aucune donnée pour {t} — vérifier le ticker.")
        continue
    df_ind = compute_indicators(df_raw, sma_windows=(sma_short, sma_long, 20))
    dfs[t] = df_ind
    valid_tickers.append(t)
    progress_bar.progress(int((i+1)/len(tickers) * 100))
progress_text.empty()
progress_bar.empty()

if not dfs:
    st.error("Aucune donnée chargée. Vérifie les tickers ou ta connexion.")
    st.stop()

# --------------- Summary KPIs (top row cards) -------------
kpi_cols = st.columns(len(valid_tickers))
summary_list = []
for idx, t in enumerate(valid_tickers):
    df = dfs[t]
    last_close = df['Close'].iloc[-1]
    ann = annualised_return(df)
    vol = annualised_vol(df)
    sr = sharpe_ratio(df)
    mdd = max_drawdown(df)
    summary_list.append({'ticker': t, 'ann': ann, 'vol': vol, 'sharpe': sr, 'mdd': mdd, 'last': last_close})
    with kpi_cols[idx]:
        arrow = "▲" if ann >= 0 else "▼"
        color = ACCENT if ann >= 0 else "#d9534f"
        st.markdown(f"<div class='kpi'><strong>{t}</strong><div class='card-num'>{arrow} {ann*100:.2f}%</div><div class='small-muted'>{vol*100:.2f}% vol · Sharpe {sr:.2f} · MDD {mdd*100:.1f}%</div></div>", unsafe_allow_html=True)

summary_df = pd.DataFrame(summary_list).set_index('ticker')
best = summary_df['ann'].idxmax()
worst = summary_df['ann'].idxmin()
st.markdown(f"**Performance résumé** — Meilleur: **{best}** · Pire: **{worst}**")

# --------------- Main area: selector + charts ----------------
left, right = st.columns([3,1])
with left:
    inspect = st.selectbox("Choisir un ticker à inspecter", valid_tickers)
    df_ins = dfs[inspect]
    # quick period selector
    period_opt = st.radio("Période rapide", ['1M','3M','6M','1Y','5Y','ALL'], index=3)
    end = df_ins.index.max()
    if period_opt == '1M':
        start = end - pd.DateOffset(months=1)
    elif period_opt == '3M':
        start = end - pd.DateOffset(months=3)
    elif period_opt == '6M':
        start = end - pd.DateOffset(months=6)
    elif period_opt == '1Y':
        start = end - pd.DateOffset(years=1)
    elif period_opt == '5Y':
        start = end - pd.DateOffset(years=5)
    else:
        start = df_ins.index.min()
    plot_df = df_ins.loc[start:end].dropna(subset=['Close'])
    # Price chart
    fig_price = plot_price_with_indicators(plot_df, inspect, sma_windows=(sma_short, sma_long, 20), show_macd=show_macd, show_bollinger=show_bollinger, show_volume=show_volume)
    st.plotly_chart(fig_price, use_container_width=True)
    # RSI
    if show_rsi:
        fig_r = plot_rsi(plot_df)
        if fig_r is not None:
            st.plotly_chart(fig_r, use_container_width=True)

    # Additional analytics (signal, backtest)
    st.markdown("### Signal & backtest (simple)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("SMA crossover signal (court vs long)")
        signal = simple_sma_signal(df_ins, short=sma_short, long=sma_long)
        latest_signal = signal.iloc[-1]
        st.write(f"Signal récent: **{'LONG' if latest_signal==1 else 'FLAT'}**")
    with col_b:
        if generate_backtest:
            # Compute simple backtest returns: buy & hold vs SMA strategy
            dfbt = df_ins[['Close']].copy()
            dfbt['Signal'] = simple_sma_signal(df_ins, short=sma_short, long=sma_long)
            dfbt['StrategyReturn'] = dfbt['Signal'].shift(1) * dfbt['Close'].pct_change().fillna(0)
            bh_cum = (1 + dfbt['Close'].pct_change().fillna(0)).cumprod()
            strat_cum = (1 + dfbt['StrategyReturn']).cumprod()
            st.write("Perf. buy&hold:", f"{(bh_cum.iloc[-1]-1)*100:.2f}%")
            st.write("Perf. SMA strategy:", f"{(strat_cum.iloc[-1]-1)*100:.2f}%")
            # small plot
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=dfbt.index, y=bh_cum, name='Buy&Hold'))
            fig_bt.add_trace(go.Scatter(x=dfbt.index, y=strat_cum, name='SMA Strategy'))
            fig_bt.update_layout(height=300, template='plotly_white', margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_bt, use_container_width=True)
    # Data table (last rows)
    st.markdown("### Données (dernières lignes)")
    nrows = st.number_input("Lignes à afficher", min_value=5, max_value=500, value=50)
    display_df = df_ins.tail(nrows)[['Open','High','Low','Close','Volume','SMA_20','SMA_50','SMA_200','EMA_20','RSI','MACD']].copy()
    # Format columns
    def format_df(df):
        df2 = df.copy()
        if 'Volume' in df2.columns:
            df2['Volume'] = df2['Volume'].apply(lambda x: f"{int(x):,}")
        for c in ['Open','High','Low','Close','SMA_20','SMA_50','SMA_200','EMA_20']:
            if c in df2.columns:
                df2[c] = df2[c].map(lambda v: f"{v:,.2f}")
        if 'RSI' in df2.columns:
            df2['RSI'] = df2['RSI'].map(lambda v: f"{v:.1f}")
        return df2
    st.dataframe(format_df(display_df))

    # Downloads for the inspected ticker
    excel_bytes = to_excel_bytes({inspect: df_ins})
    st.download_button("⬇️ Télécharger Excel (ticker)", data=excel_bytes, file_name=f"{inspect}_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    csv_bytes = df_ins.to_csv().encode('utf-8')
    st.download_button("⬇️ Télécharger CSV (ticker)", data=csv_bytes, file_name=f"{inspect}_data.csv", mime='text/csv')
    if st.button("Générer rapport PDF (résumé)"):
        pdf_bytes = generate_pdf_report(inspect, df_ins, plotly_fig=fig_price)
        st.download_button("⬇️ Télécharger PDF", data=pdf_bytes, file_name=f"{inspect}_report.pdf", mime='application/pdf')

with right:
    st.markdown("### Comparaisons & Heatmap")
    # Correlation heatmap
    heat = plot_correlation_heatmap(dfs)
    if heat is not None:
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Heatmap non disponible (données insuffisantes)")

    st.markdown("### Top / Flop (annualised return)")
    sorted_df = summary_df.sort_values('ann', ascending=False)
    st.table(sorted_df[['ann','vol','sharpe']].apply(lambda x: x.map(lambda v: f"{v:.2f}" if isinstance(v, (int,float)) and not pd.isna(v) else v)))

    st.markdown("### Benchmark (optionnel)")
    if benchmark and benchmark != 'None':
        bdf = fetch_ticker_history(benchmark, pd.to_datetime(start_date), pd.to_datetime(end_date)+pd.Timedelta(days=1))
        if bdf is not None:
            m = pd.DataFrame({inspect: df_ins['Close'], benchmark: bdf['Close']}).dropna()
            if not m.empty:
                m_norm = m / m.iloc[0] * 100
                figb = px.line(m_norm, labels={'value':'Indexed (100)','index':'Date'})
                st.plotly_chart(figb, use_container_width=True)
            else:
                st.warning("Benchmark non comparable (période).")
        else:
            st.warning("Impossible de charger le benchmark.")

# --------------- News (optional) ----------------
st.markdown('---')
st.markdown("## Actualités & Sentiment (optionnel)")
if news_api_key:
    try:
        q = f"{inspect} bank OR banking OR finance"
        url = f"https://newsapi.org/v2/everything?q={requests.utils.requote_uri(q)}&pageSize=5&sortBy=publishedAt&language=fr&apiKey={news_api_key}"
        r = requests.get(url, timeout=10)
        j = r.json()
        arts = j.get('articles', [])
        if not arts:
            st.info("Aucun article récent.")
        for a in arts:
            published = a.get('publishedAt')
            dt = pd.to_datetime(published) if published else None
            st.markdown(f"**{a.get('title')}** — <span class='small-muted'>{dt.date() if dt else ''}</span>", unsafe_allow_html=True)
            st.write(a.get('description') or a.get('content') or "")
            st.markdown(f"[Lire l'article]({a.get('url')})")
            st.markdown('---')
    except Exception as e:
        st.error(f"Erreur news: {e}")
else:
    st.info("Renseigne une clé NewsAPI dans la barre latérale pour activer les actualités (optionnel).")

# --------------- Portfolio builder (simple) ----------------
st.markdown("---")
st.markdown("## Portefeuille simple (simulateur)")
with st.expander("Configurer un portefeuille / comparer allocations"):
    cols = st.columns(2)
    with cols[0]:
        selected_tickers = st.multiselect("Tickers (portefeuille)", valid_tickers, default=valid_tickers[:3])
        weights_text = st.text_input("Poids (séparés par virgule, en décimal ex: 0.5,0.3,0.2)", value=",".join(["{:.2f}".format(1/len(selected_tickers))]*len(selected_tickers)) if selected_tickers else "")
    with cols[1]:
        initial_cap = st.number_input("Capital initial (€)", value=10000.0, step=100.0)
        rebalance = st.selectbox("Rebalancer", ["None","Monthly","Quarterly","Yearly"], index=0)
    if st.button("Calculer portefeuille"):
        if not selected_tickers:
            st.warning("Sélectionne au moins un ticker.")
        else:
            try:
                weights = [float(w.strip()) for w in weights_text.split(',')]
                if len(weights) != len(selected_tickers):
                    st.error("Le nombre de poids doit correspondre aux tickers.")
                else:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    # Build aligned returns
                    closes = pd.DataFrame({t: dfs[t]['Close'] for t in selected_tickers}).dropna()
                    rets = closes.pct_change().dropna()
                    port_ret = (rets * weights).sum(axis=1)
                    port_cum = (1 + port_ret).cumprod()
                    figp = go.Figure()
                    figp.add_trace(go.Scatter(x=port_cum.index, y=port_cum * initial_cap, name='Portefeuille Value'))
                    st.plotly_chart(figp, use_container_width=True)
                    st.write("Perf totale:", f"{(port_cum.iloc[-1]-1)*100:.2f}%")
                    st.write("Volatility annualisée:", f"{port_ret.std()*np.sqrt(252)*100:.2f}%")
            except Exception as e:
                st.error("Erreur calcul portefeuille: " + str(e))

# --------------- Exports multi-ticker ----------------
st.markdown('---')
st.markdown("### Export multi-tickers")
if st.button("Télécharger données pour tous les tickers (Excel)"):
    all_bytes = to_excel_bytes({k: v for k,v in dfs.items()})
    st.download_button("⬇️ Télécharger Excel (tous)", data=all_bytes, file_name="all_tickers_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------- Footer & credibility ----------------
st.markdown('---')
st.markdown(f"<div class='small-muted'>Sources: Yahoo Finance via yfinance. Données fournies à titre informatif — vérifier avant toute décision d'investissement.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# End of file
