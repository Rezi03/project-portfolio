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
import time
import matplotlib.pyplot as plt

st.cache_data.clear()

st.set_page_config(
    page_title="Global Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NOUVEAU : Dictionnaire groupé pour tous les onglets ---
TICKERS_GROUPED = {
    "Banques & Finance": {
        # Amériques
        "J.P. Morgan Chase (US)": "JPM",
        "Bank of America (US)": "BAC",
        "Wells Fargo (US)": "WFC",
        "CitiGroup (US)": "C",
        "Goldman Sachs (US)": "GS",
        "Morgan Stanley (US)": "MS",
        "PNC Financial (US)": "PNC",
        "Royal Bank of Canada (CA)": "RY.TO",
        # Europe
        "UBS Group (CH)": "UBSG.SW",
        "BNP Paribas (FR)": "BNP.PA",
        "Credit Agricole (FR)": "ACA.PA",
        "Deutsche Bank (DE)": "DBK.DE",
        "Santander (ES)": "SAN.MC",
        "UniCredit (IT)": "UCG.MI",
        "Barclays (UK)": "BARC.L",
        "HSBC Holdings (UK/HK)": "HSBA.L",
        # Asie
        "Mizuho Financial (JP)": "8411.T",
        "Mitsubishi UFJ (JP)": "8306.T",
    },
    "Tech & Croissance": {
        "Apple Inc.": "AAPL",
        "Microsoft Corp.": "MSFT",
        "Alphabet Inc. (Google)": "GOOGL",
        "Amazon.com Inc.": "AMZN",
        "Meta Platforms Inc.": "META",
        "Tesla Inc.": "TSLA",
        "ASML Holding (NL)": "ASML",
        "Tencent (HK)": "0700.HK",
    },
    "Matières Premières & Énergie": {
        "Gold (Futures)": "GC=F",
        "Crude Oil (WTI)": "CL=F",
        "Exxon Mobil": "XOM",
        "Chevron Corp.": "CVX",
        "BHP Group (AU)": "BHP.AX",
    },
    "Crypto (Via Indices/Trusts)": {
        "Bitcoin (USD)": "BTC-USD",
        "Ethereum (USD)": "ETH-USD",
        "Coinbase Global": "COIN",
        "MicroStrategy": "MSTR",
    },
    "Macro & Indices": {
        "S&P 500": "^GSPC",
        "NASDAQ Composite": "^IXIC",
        "Euro Stoxx 50": "^STOXX50E",
        "Nikkei 225": "^N225",
        "DAX Index": "^GDAXI",
        "US 10 Year Yield": "^TNX",
    }
}
# La référence pour le BETA reste le S&P 500, utilisé dans l'onglet Macro & Indices.
BENCHMARK = "^GSPC" 

# NEWS_API_KEY est retiré comme demandé
# ... (le reste des fonctions utilitaires reste le même) ...
# Je ne recopie pas les fonctions fetch_history, sma, compute_rsi, annualized_return, cagr, etc. 
# car elles n'ont pas été modifiées.

@st.cache_data(ttl=0, show_spinner=False)
def fetch_history(ticker, period="1y", interval="1d"):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Date": "Date"})
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_info(ticker):
    try:
        return yf.Ticker(ticker).info
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
    return 100 - (100 / (1 + rs))

def annualized_return(df):
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        return np.nan
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
    if returns.empty:
        return np.Series(dtype=float)
    return returns.rolling(window).std() * np.sqrt(252)

def sharpe_ratio(df, risk_free_rate=0.02):
    returns = df['Close'].pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_return = returns - (risk_free_rate / 252)
    return np.sqrt(252) * (excess_return.mean() / excess_return.std())

def sortino_ratio(df, risk_free_rate=0.02, period=252):
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        return np.nan
    excess = returns - (risk_free_rate / period)
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        return np.nan
    downside_std = negative_returns.std()
    if downside_std == 0:
        return np.nan
    return (np.sqrt(period) * excess.mean()) / downside_std

def beta_and_vol(df, benchmark_df):
    try:
        returns_stock = df['Close'].pct_change().dropna()
        returns_bench = benchmark_df['Close'].pct_change().dropna()
        joined = pd.concat([returns_stock, returns_bench], axis=1).dropna()
        if joined.shape[0] < 2:
            return np.nan, np.nan
        cov = joined.cov()
        beta = cov.iloc[0,1] / cov.iloc[1,1] if cov.iloc[1,1] != 0 else np.nan
        volatility = returns_stock.std() * np.sqrt(252)
        return beta, volatility
    except Exception:
        return np.nan, np.nan

def value_at_risk(df, confidence=0.95):
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        return np.nan
    return -np.percentile(returns, (1 - confidence) * 100)

def alpha_annualized(df, benchmark_df, risk_free_rate=0.02):
    try:
        r_stock = df['Close'].pct_change().dropna()
        r_bench = benchmark_df['Close'].pct_change().dropna()
        joined = pd.concat([r_stock, r_bench], axis=1).dropna()
        if joined.shape[0] < 2:
            return np.nan
        y = joined.iloc[:,0] - (risk_free_rate / 252)
        x = joined.iloc[:,1] - (risk_free_rate / 252)
        X = np.vstack([np.ones(len(x)), x]).T
        coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        intercept = coef[0]
        alpha_ann = intercept * 252
        return alpha_ann
    except Exception:
        return np.nan

def simulate_investment(df, initial_capital=1.0):
    returns = df['Close'].pct_change().fillna(0)
    equity = (1 + returns).cumprod()
    return equity

def generate_alerts(df):
    alerts = []
    try:
        if 'RSI' not in df.columns:
            df['RSI'] = compute_rsi(df['Close'])
        if df['RSI'].iloc[-1] >= 70:
            alerts.append("RSI ≥ 70 → Overbought possible")
        if df['RSI'].iloc[-1] <= 30:
            alerts.append("RSI ≤ 30 → Oversold possible")

        if 'SMA200' not in df.columns:
            df['SMA200'] = sma(df['Close'], 200)
        if df['Close'].iloc[-1] < df['SMA200'].iloc[-1]:
            alerts.append("Price below SMA200 → bearish long-term signal")

        vol = rolling_volatility(df, window=21)
        if not vol.empty and vol.iloc[-1] > 0.6: 
            alerts.append("High annualized volatility (>60%)")
    except Exception:
        pass
    return alerts

def create_placeholder_png(message="Chart not available"):
    buf = BytesIO()
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def render_chart_bytes(name, fig_obj=None, df=None, title="Chart"):
    # (J'omets le corps de cette fonction car il est très long et inchangé)
    # ... (Le code de render_chart_bytes reste le même) ...
    if fig_obj is not None:
        try:
            # Assurez-vous d'avoir 'kaleido' installé pour la conversion Plotly -> PNG
            # pip install kaleido
            import kaleido  
            try:
                img_bytes = fig_obj.to_image(format="png")
                if img_bytes:
                    return img_bytes
            except Exception:
                pass
        except Exception:
            pass

    if df is None or df.empty:
        return create_placeholder_png(f"{title} - no data")

    # (Le reste du corps de render_chart_bytes est omis pour la concision)
    # ...
    try:
        buf = BytesIO()
        x = df['Date']
        if name == "price":
            plt.figure(figsize=(10,4))
            plt.plot(x, df['Close'], label='Close', linewidth=1.4)
            if 'SMA50' not in df.columns:
                df['SMA50'] = sma(df['Close'], 50)
            if 'SMA200' not in df.columns:
                df['SMA200'] = sma(df['Close'], 200)
            if df['SMA50'].notna().any():
                plt.plot(x, df['SMA50'], label='SMA50', linestyle='--', linewidth=1)
            if df['SMA200'].notna().any():
                plt.plot(x, df['SMA200'], label='SMA200', linestyle=':', linewidth=1)
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

        if name == "rsi":
            if 'RSI' not in df.columns:
                df['RSI'] = compute_rsi(df['Close'])
            plt.figure(figsize=(10,3))
            plt.plot(x, df['RSI'], label='RSI (14)')
            plt.axhline(70, color='red', linestyle='--', linewidth=0.7)
            plt.axhline(30, color='green', linestyle='--', linewidth=0.7)
            plt.title(title)
            plt.ylim(0, 100)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

        if name == "vol":
            if 'RollingVol21' not in df.columns:
                df['RollingVol21'] = rolling_volatility(df, window=21)
            plt.figure(figsize=(10,3))
            plt.plot(x, df['RollingVol21'], label='21-day rolling vol')
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

        if name == "equity":
            eq = simulate_investment(df)
            plt.figure(figsize=(10,3))
            plt.plot(x, eq, label='Equity Curve')
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

        if name == "hist":
            returns = df['Close'].pct_change().dropna()
            plt.figure(figsize=(8,3))
            plt.hist(returns, bins=50)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

        return create_placeholder_png(f"{name} - chart type not handled")
    except Exception as e:
        return create_placeholder_png(f"Render error: {str(e)[:80]}")
    # ...

def generate_analysis(df, metrics, rf_rate=0.02):
    # (Le corps de cette fonction est long et inchangé)
    # ... (Le code de generate_analysis reste le même) ...
    def fmt_ratio(x):
        try:
            return f"{x:.2f}"
        except Exception:
            return "N/A"
    def fmt_pct(x):
        try:
            return f"{x*100:.2f}%"
        except Exception:
            return "N/A"

    annual_ret = metrics.get("annual_ret", np.nan)
    vol = metrics.get("vol", np.nan)
    sharpe = metrics.get("sharpe", np.nan)
    sortino = metrics.get("sortino", np.nan)
    beta = metrics.get("beta", np.nan)
    alpha = metrics.get("alpha", np.nan)
    var95 = metrics.get("var95", np.nan)
    cagr_val = metrics.get("cagr", np.nan)
    max_dd = metrics.get("max_dd", np.nan)

    bullets = []
    score = 0

    try:
        if 'SMA50' in df.columns and 'SMA200' in df.columns:
            if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]:
                bullets.append("SMA50 > SMA200 → medium-term bullish trend.")
                score += 1
            else:
                bullets.append("SMA50 < SMA200 → medium-term bearish trend.")
                score -= 1
    except Exception:
        pass

    try:
        if 'RSI' in df.columns:
            rsi_now = df['RSI'].iloc[-1]
            if rsi_now >= 70:
                bullets.append(f"RSI ({rsi_now:.1f}) high → potential overbought condition.")
                score -= 1
            elif rsi_now <= 30:
                bullets.append(f"RSI ({rsi_now:.1f}) low → potential oversold rebound.")
                score += 1
            else:
                bullets.append(f"RSI ({rsi_now:.1f}) neutral.")
    except Exception:
        pass

    try:
        if not np.isnan(vol):
            if vol > 0.6:
                bullets.append(f"High annualized volatility ({vol*100:.1f}%) → elevated risk.")
                score -= 1
            elif vol < 0.15:
                bullets.append(f"Low annualized volatility ({vol*100:.1f}%) → calm market.")
                score += 1
            else:
                bullets.append(f"Moderate annualized volatility ({vol*100:.1f}%).")
    except Exception:
        pass

    try:
        if not np.isnan(sharpe):
            if sharpe > 1:
                bullets.append(f"Sharpe ratio strong ({fmt_ratio(sharpe)}) → good risk-adjusted returns.")
                score += 1
            elif sharpe < 0.5:
                bullets.append(f"Sharpe ratio weak ({fmt_ratio(sharpe)}) → low return per unit risk.")
                score -= 1
            else:
                bullets.append(f"Sharpe ratio moderate ({fmt_ratio(sharpe)}).")
    except Exception:
        pass

    try:
        if not np.isnan(sortino):
            bullets.append(f"Sortino ratio: {fmt_ratio(sortino)} (focuses on downside risk).")
    except Exception:
        pass

    try:
        if not np.isnan(var95):
            bullets.append(f"VaR 95%: {fmt_pct(var95)}.")
        if not np.isnan(max_dd):
            bullets.append(f"Max Drawdown: {fmt_pct(max_dd)} for the analyzed period.")
            if max_dd < -0.3:
                bullets.append("Drawdown > 30% historically → caution advised.")
                score -= 1
    except Exception:
        pass

    try:
        if not np.isnan(beta):
            bullets.append(f"Beta vs benchmark: {fmt_ratio(beta)} (market sensitivity).")
        if not np.isnan(alpha):
            bullets.append(f"Alpha (annualized): {fmt_pct(alpha)} (risk-adjusted outperformance).")
            if alpha > 0.02:
                bullets.append("Positive alpha >2% annualized → potential value-add.")
                score += 1
    except Exception:
        pass

    try:
        if not np.isnan(cagr_val):
            bullets.append(f"CAGR: {fmt_pct(cagr_val)}.")
    except Exception:
        pass

    recs = []
    if score >= 2:
        recs.append("Overall signal: Positive — consider cautious allocation.")
    elif score <= -2:
        recs.append("Overall signal: Negative — consider reducing exposure or hedging.")
    else:
        recs.append("Overall signal: Neutral — monitor indicators for confirmation.")

    parts = []
    parts.append(f"Over the selected period the annualized return is {fmt_pct(annual_ret)} and annualized volatility is {fmt_pct(vol)}.")
    if not np.isnan(sharpe):
        parts.append(f"Sharpe: {fmt_ratio(sharpe)}.")
    if not np.isnan(beta):
        parts.append(f"Beta vs benchmark: {fmt_ratio(beta)}.")
    if not np.isnan(alpha):
        parts.append(f"Alpha (annualized): {fmt_pct(alpha)}.")
    parts.append(recs[0])

    analysis_text = " ".join(parts)
    analysis_bullets = bullets + [""] + recs
    return analysis_text, analysis_bullets
    # ...

# --- NOUVELLE FONCTION : Regrouper la logique du Dashboard ---
def render_dashboard(ticker_dict, tab_name):
    
    st.markdown(f"<h1 style='font-family:-apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto;'>{tab_name} Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # Récupérer la liste des actifs pour cet onglet
    asset_list = list(ticker_dict.keys())
    
    mode = st.sidebar.radio("Mode", ["Analyse Détaillée", "Comparaison"], key=f"mode_{tab_name}")

    if mode == "Analyse Détaillée":
        bank = st.sidebar.selectbox(f"Sélectionner {tab_name.split(' & ')[0]}", asset_list, key=f"select_{tab_name}")
        ticker = ticker_dict[bank]
        
        # Le code d'analyse détaillée est réutilisé ici
        df = fetch_history(ticker, period, interval)
        info = fetch_info(ticker)

        if df.empty or 'Close' not in df.columns or df['Close'].isnull().all():
            st.error(f"Aucune donnée de marché disponible pour {bank} et cette période.")
        else:
            benchmark_df = fetch_history(BENCHMARK, period, interval)

            if show_tech:
                df['SMA50'] = sma(df['Close'], 50)
                df['SMA200'] = sma(df['Close'], 200)
                df['RSI'] = compute_rsi(df['Close'])
                df['RollingVol21'] = rolling_volatility(df, window=21)

            last_price = df['Close'].iloc[-1]
            market_cap = info.get('marketCap', None)
            cagr_val = cagr(df)
            max_dd = max_drawdown(df)
            sharpe = sharpe_ratio(df, risk_free_rate=rf_rate)
            sortino = sortino_ratio(df, risk_free_rate=rf_rate)
            var95 = value_at_risk(df, confidence=0.95)
            annual_ret = annualized_return(df)
            beta, vol = beta_and_vol(df, benchmark_df)
            alpha_ann = alpha_annualized(df, benchmark_df, risk_free_rate=rf_rate)
            treynor = (annual_ret - rf_rate) / beta if (not np.isnan(annual_ret) and not np.isnan(beta) and beta != 0) else np.nan

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Dernier Prix (USD)", f"{last_price:.2f}")
            col2.metric("Cap. Marché", f"{market_cap:,}" if market_cap else "N/A")
            col3.metric("CAGR (annuel)", f"{cagr_val*100:.2f}%" if not np.isnan(cagr_val) else "N/A")
            col4.metric("Max Drawdown", f"{max_dd*100:.2f}%" if not np.isnan(max_dd) else "N/A")
            col5.metric("Ratio de Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")

            col6, col7, col8, col9, col10 = st.columns(5)
            col6.metric("Ratio de Sortino", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")
            col7.metric("Volatilité Ann.", f"{vol*100:.2f}%" if not np.isnan(vol) else "N/A")
            col8.metric("Beta vs S&P500", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
            col9.metric("Alpha (ann.)", f"{alpha_ann*100:.2f}%" if not np.isnan(alpha_ann) else "N/A")
            col10.metric("Ratio de Treynor", f"{treynor:.2f}" if not np.isnan(treynor) else "N/A")

# --- BLOC GRAPHIQUE "WIPE EFFECT" (PROPRE & FLUIDE) ---
            
            # 1. CSS pour l'effet "Dessin" (Balayage gauche -> droite)
            st.markdown(f"""
                <style>
                /* On cible le graphique spécifique avec une animation de masque */
                div[data-testid="stPlotlyChart"] > div {{
                    animation: wipeEnter 1.2s cubic-bezier(0.22, 1, 0.36, 1) both;
                }}
                
                @keyframes wipeEnter {{
                    0% {{ 
                        clip-path: inset(0 100% 0 0); /* Masqué à 100% à droite */
                        opacity: 0.5;
                    }}
                    100% {{ 
                        clip-path: inset(0 0 0 0);    /* Visible entièrement */
                        opacity: 1;
                    }}
                }}
                </style>
            """, unsafe_allow_html=True)

            # 2. Création du graphique
            fig_price = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Prix'
            )])

            # 3. Design Épuré
            fig_price.update_layout(
                title=dict(text=f"{bank}", font=dict(size=24)),
                yaxis_title="Prix (USD)",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=550,
                margin=dict(l=10, r=10, t=50, b=20),
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, range=[df['Date'].iloc[0], df['Date'].iloc[-1]]),
                yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
            )
            
            # 4. AFFICHAGE AVEC CLÉ UNIQUE (C'est le secret pour relancer l'anim)
            # On ajoute key=bank pour forcer Streamlit à recréer le bloc à chaque changement d'actif
            st.plotly_chart(fig_price, use_container_width=True, key=f"chart_{ticker}_{period}")
            
            # --- FIN BLOC ---

            left, right = st.columns([2,1])
            with left:
                if show_tech:
                    fig_sma = go.Figure()
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Clôture', mode='lines'))
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name='SMA50', mode='lines'))
                    fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], name='SMA200', mode='lines'))
                    fig_sma.update_layout(title="Prix avec Moyennes Mobiles", template="plotly_white", height=350)
                    st.plotly_chart(fig_sma, use_container_width=True)

                    fig_rsi = px.line(df, x='Date', y='RSI', title="RSI (14)", template="plotly_white", height=220)
                    fig_rsi.update_yaxes(range=[0,100])
                    st.plotly_chart(fig_rsi, use_container_width=True)

            with right:
                fig_hist = px.histogram(df.assign(Returns=df['Close'].pct_change()), x='Returns', nbins=50, title="Distribution des Rendements Quotidiens", template="plotly_white", height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

                equity = simulate_investment(df)
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=df['Date'], y=equity, name='Courbe d\'Équité'))
                fig_eq.update_layout(title="Courbe d'Équité (1 unité investie)", template="plotly_white", height=300)
                st.plotly_chart(fig_eq, use_container_width=True)

                alerts = generate_alerts(df)
                if alerts:
                    st.warning(" | ".join(alerts))

            if 'RollingVol21' not in df.columns:
                df['RollingVol21'] = rolling_volatility(df, window=21)
            fig_roll = px.line(df, x='Date', y='RollingVol21', title="Volatilité Glissante 21 jours (annualisée)", template="plotly_white", height=220)
            st.plotly_chart(fig_roll, use_container_width=True)

            metrics_for_analysis = {
                "annual_ret": annual_ret,
                "vol": vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "beta": beta,
                "alpha": alpha_ann,
                "var95": var95,
                "cagr": cagr_val,
                "max_dd": max_dd
            }

            analysis_text, analysis_bullets = generate_analysis(df, metrics_for_analysis, rf_rate=rf_rate)

            st.markdown("**Synthèse numérique automatisée:**")
            summary_lines = []
            if not np.isnan(annual_ret):
                summary_lines.append(f"Rend. Annuel: {annual_ret*100:.2f}%")
            if not np.isnan(vol):
                summary_lines.append(f"Vol. Annuelle: {vol*100:.2f}%")
            if not np.isnan(sharpe):
                summary_lines.append(f"Sharpe: {sharpe:.2f}")
            if not np.isnan(sortino):
                summary_lines.append(f"Sortino: {sortino:.2f}")
            if not np.isnan(beta):
                summary_lines.append(f"Beta vs S&P500: {beta:.2f}")
            if not np.isnan(alpha_ann):
                summary_lines.append(f"Alpha (ann.): {alpha_ann*100:.2f}%")
            if not np.isnan(var95):
                summary_lines.append(f"VaR 95%: {var95*100:.2f}%")
            auto_text = " | ".join(summary_lines)
            st.code(auto_text, language="text")

            st.markdown("**Analyse automatisée (basée sur règles):**")
            for b in analysis_bullets:
                if b:
                    st.write("- " + b)

            # ... (Logique de génération de PDF omise pour la concision) ...
            # C'est la partie qui nécessite kaleido et fpdf

    else: # Mode Comparaison
        banks = st.sidebar.multiselect(f"Sélectionner les actifs à comparer ({tab_name})", asset_list, default=asset_list[:2], key=f"multi_{tab_name}")
        
        if len(banks) < 2:
            st.warning("Sélectionnez au moins deux actifs pour la comparaison.")
        else:
            data = {}
            for b in banks:
                data[b] = fetch_history(ticker_dict[b], period, interval)

            fig = go.Figure()
            for bank, dfb in data.items():
                if not dfb.empty and 'Close' in dfb.columns:
                    dfb_sorted = dfb.sort_values('Date')
                    cum = (1 + dfb_sorted['Close'].pct_change().fillna(0)).cumprod()
                    fig.add_trace(go.Scatter(x=dfb_sorted['Date'], y=cum, mode='lines', name=bank))
            fig.update_layout(title=f"Rendements Cumulés Normalisés ({tab_name})", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            metrics = []
            benchmark_df = fetch_history(BENCHMARK, period, interval)
            for b in banks:
                dfb = data[b]
                if dfb.empty:
                    metrics.append({"Actif": b, "CAGR": "N/A", "Sharpe": "N/A", "Volatilité": "N/A", "Beta": "N/A"})
                    continue
                
                beta_b, vol_b = beta_and_vol(dfb, benchmark_df)
                metrics.append({
                    "Actif": b,
                    "CAGR": f"{cagr(dfb)*100:.2f}%" if not np.isnan(cagr(dfb)) else "N/A",
                    "Sharpe": f"{sharpe_ratio(dfb, risk_free_rate=rf_rate):.2f}" if not np.isnan(sharpe_ratio(dfb, risk_free_rate=rf_rate)) else "N/A",
                    "Volatilité": f"{vol_b*100:.2f}%" if not np.isnan(vol_b) else "N/A",
                    "Beta": f"{beta_b:.2f}" if not np.isnan(beta_b) else "N/A"
                })
            st.dataframe(pd.DataFrame(metrics))

            returns_df = pd.DataFrame()
            for b in banks:
                dfb = data[b].set_index('Date').sort_index()
                if 'Close' in dfb.columns:
                    returns_df[b] = dfb['Close'].pct_change()
            if returns_df.dropna(how='all').shape[1] > 1:
                corr = returns_df.corr()
                fig_corr = px.imshow(corr, text_auto=".2f", title="Matrice de Corrélation des Rendements", template="plotly_white")
                st.plotly_chart(fig_corr, use_container_width=True)


# --- LOGIQUE PRINCIPALE DE STREAMLIT ---
st.sidebar.title("Contrôles")
# --- NOUVEAU : Onglets Dynamiques ---
tabs = list(TICKERS_GROUPED.keys())
tabs.insert(0, "Sélectionner un Onglet") # Ajout d'un placeholder
tab = st.sidebar.selectbox("Sélectionnez la Catégorie", tabs)
st.sidebar.markdown("---")
st.sidebar.caption("Construit avec Streamlit, yfinance, Plotly")

# Contrôles globaux
period = st.sidebar.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)
show_tech = st.sidebar.checkbox("Afficher les indicateurs techniques", value=True)
rf_rate = st.sidebar.number_input("Taux sans risque (annuel, ex. 0.02)", value=0.02, step=0.005, format="%.4f")
st.sidebar.markdown("---")

if tab == "Sélectionner un Onglet":
    st.markdown("<h1>Bienvenue sur le Global Market Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.info("Veuillez sélectionner une catégorie (Banques, Tech, Crypto, etc.) dans la barre latérale pour commencer l'analyse.")
else:
    # Appel à la fonction de rendu du tableau de bord pour l'onglet sélectionné
    render_dashboard(TICKERS_GROUPED[tab], tab)