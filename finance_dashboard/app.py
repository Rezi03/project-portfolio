# app.py — Hub d’accueil
import streamlit as st
from utils.ga import inject_ga4

st.set_page_config(
    page_title="Finance Projects Hub — IB Track",
    page_icon="💼",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Interactive finance portfolio tailored for Investment Banking applications (GS/MS/JPM/BoA)."
    }
)

# GA4 tracking
inject_ga4()

# ---- STYLE LÉGER
st.markdown(
    """
    <style>
      .big-title {font-size:2.1rem; font-weight:700; margin-bottom:0.2rem;}
      .sub {color:#666; margin-top:0; font-size:0.95rem;}
      .card {
        padding: 1.0rem 1.2rem; border:1px solid #e7e7e7; border-radius:14px;
        background: rgba(250,250,250,0.7);
      }
      .muted { color:#6b7280; }
      .kbd {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#f1f5f9; padding:.15rem .35rem; border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- HEADER
st.markdown('<div class="big-title">Finance Projects — Investment Banking Track</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">M&A quasi temps réel • DCF interactif • Backtester Actions/Obligations • GA4 usage tracking</p>', unsafe_allow_html=True)

# ---- SECTION: NAVIGATION RAPIDE
st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 📈 M&A Deals Dashboard")
    st.markdown(
        "Flux FMP, filtres par date/secteur/valeur, timeline interactive, news. "
        "Montre ta capacité à suivre l’activité de marché quasi en temps réel."
    )
    st.page_link("pages/1_M&A_Deals_Dashboard.py", label="Ouvrir M&A Dashboard →")

with c2:
    st.markdown("### 💡 DCF Lab")
    st.markdown(
        "Valorisation par flux actualisés (FCF), WACC, terminal value, sensibilité g/WACC, export PDF. "
        "Montre ta compréhension des fondamentaux IB."
    )
    st.page_link("pages/2_DCF_Lab.py", label="Ouvrir DCF Lab →")

with c3:
    st.markdown("### 🔎 Backtester Actions/Obligations")
    st.markdown(
        "Télécharge des séries (yfinance), métriques de perf/risque (CAGR, Sharpe, MaxDD), comparaison benchmark. "
        "Montre rigueur et culture marché."
    )
    st.page_link("pages/3_Backtester_Equity_Bonds.py", label="Ouvrir Backtester →")

st.write("")
st.divider()

# ---- SECTION: CONSEILS CANDIDATURE IB
st.markdown("#### 🎯 Pourquoi c’est pertinent pour l’IB (ex. Goldman Sachs)")
st.markdown(
    "- **Orientation deal-driven** : Monitoring M&A live + news.\n"
    "- **Valo** : DCF propre, hypothèses transparentes, sensis, export rapport.\n"
    "- **Rigueur marchés** : Backtests clairs, lecture risque/rendement.\n"
    "- **Data/Produit** : App web structurée, traçable (GA4), réutilisable en entretien.\n"
)

with st.expander("📌 Tips pour CV/entretiens (à recycler dans tes bullet points)"):
    st.markdown(
        """
        - Built a **real-time M&A monitoring** app using FMP API, **filtered by sector/region/value**, with interactive visuals and news aggregation.
        - Designed a **DCF valuation lab** with scenario analysis and **PDF export** for investment memos.
        - Implemented an **equity/bond backtester** with performance & risk metrics (CAGR, Sharpe, Max Drawdown), **benchmarking** and parameter controls.
        - Instrumented **GA4 analytics** to track feature usage and identify the most impactful modules.
        """
    )

st.write("")
st.caption("Astuce : si le bouton `Ouvrir →` n’apparaît pas, utilise le menu **Pages** dans la sidebar Streamlit.")

# ---- FOOTER D'INFOS
st.write("")
st.markdown(
    '<div class="card">'
    '<span class="muted">Sécurité :</span> Les clés (FMP, NewsAPI, GA4) sont stockées dans <span class="kbd">.streamlit/secrets.toml</span> '
    "ou dans les <em>Secrets</em> de Streamlit Cloud. Aucun secret n’est commité sur GitHub."
    "</div>",
    unsafe_allow_html=True
)
