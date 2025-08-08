import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from fpdf import FPDF
from io import BytesIO
import numpy as np

# =========================
# CONFIGURATION DE LA PAGE
# =========================
st.set_page_config(
    page_title="Banking Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# LIRE LA CLÉ API DEPUIS SECRETS
# =========================
NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY", None)

# =========================
# FONCTIONS UTILES
# =========================
def sharpe_ratio(df):
    """Calcule le Sharpe ratio annuel"""
    daily_return = df['Close'].pct_change()
    return (daily_return.mean() / daily_return.std()) * np.sqrt(252)

def get_news(query):
    """Récupère les news depuis NewsAPI"""
    if not NEWS_API_KEY:
        st.info("ℹ️ NewsAPI non configuré — les actualités ne seront pas affichées.")
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    res = requests.get(url, timeout=6).json()
    return res.get("articles", [])

def generate_pdf_report(bank, ticker, df, info, beta):
    """Génère un PDF avec les principales métriques"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, f"{bank} - Financial Report", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Last Price: {df['Close'].iloc[-1]:.2f} USD", ln=True)
    pdf.cell(0, 10, f"Market Cap: {info.get('marketCap', 'N/A'):,}", ln=True)
    pdf.cell(0, 10, f"Beta: {beta:.2f} | Sharpe: {sharpe_ratio(df):.2f}", ln=True)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return BytesIO(pdf_bytes)

# =========================
# INTERFACE PRINCIPALE
# =========================
st.title("📊 Banking Market Intelligence Dashboard")
st.markdown("Analyse en direct des banques avec données boursières, indicateurs et actualités.")

banks = {
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "J.P. Morgan": "JPM",
    "Citigroup": "C",
    "Bank of America": "BAC",
    "Barclays": "BCS"
}

bank = st.selectbox("Sélectionnez une banque :", list(banks.keys()))
ticker = banks[bank]

# Téléchargement des données
df = yf.download(ticker, period="6mo", interval="1d")
info = yf.Ticker(ticker).info
beta = info.get("beta", 0)

# =========================
# ONGLET 1 : MARKET DATA
# ONGLET 2 : NEWS
# =========================
tabs = st.tabs(["📈 Market Data", "📰 Latest News"])

with tabs[0]:
    # Graphique des prix
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    ))
    fig.update_layout(title=f"{bank} - Cours sur 6 mois", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Stats clés
    col1, col2, col3 = st.columns(3)
    col1.metric("Dernier prix", f"{df['Close'].iloc[-1]:.2f} USD")
    col2.metric("Market Cap", f"{info.get('marketCap', 0):,}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio(df):.2f}")

    # Export CSV
    st.download_button(
        label="📥 Télécharger CSV",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_data.csv",
        mime="text/csv"
    )

    # Génération PDF
    pdf_buffer = generate_pdf_report(bank, ticker, df, info, beta)
    st.download_button(
        label="📄 Télécharger PDF Report",
        data=pdf_buffer,
        file_name=f"{ticker}_report.pdf",
        mime="application/pdf"
    )

with tabs[1]:
    st.subheader(f"📰 Dernières actualités - {bank}")
    articles = get_news(bank)
    if not articles:
        st.warning("Aucune actualité trouvée ou API non configurée.")
    else:
        for art in articles:
            st.markdown(f"**[{art.get('title')}]({art.get('url')})** — *{art.get('source', {}).get('name', '')}*")
            st.write(art.get("description", ""))
            st.markdown("---")
