import streamlit as st
from utils.ga import inject_ga4
from utils.news import fetch_news

st.set_page_config(
    page_title="Finance Projects — IB Track",
    layout="wide",
)

# GA4 tracking (silencieux)
inject_ga4()

# ---------- Global style (CSS minimal, typographie et alignements) ----------
st.markdown(
    """
    <style>
      :root {
        --bg: #ffffff;
        --fg: #0f172a;
        --muted: #6b7280;
        --border: #e5e7eb;
        --card: #fafafa;
      }
      .app-container { max-width: 1200px; margin: 0 auto; }
      .title { font-size: 28px; font-weight: 700; letter-spacing: 0.2px; margin: 8px 0 2px; color: var(--fg); }
      .subtitle { font-size: 14px; color: var(--muted); margin: 0 0 24px; }
      .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
      .card {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 16px;
        padding: 18px;
        transition: transform .12s ease, box-shadow .12s ease;
      }
      .card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.06); }
      .card-title { font-size: 18px; font-weight: 600; margin: 4px 0 4px; }
      .card-desc { font-size: 13px; color: var(--muted); margin: 0 0 10px; min-height: 40px; }
      .btn-row { display: flex; gap: 8px; align-items: center; }
      .btn { border: 1px solid var(--border); padding: 8px 12px; border-radius: 10px; text-decoration: none; font-size: 13px; }
      .btn-link { color: #111827; background: #fff; }
      .section-title { font-size: 18px; font-weight: 600; margin: 24px 0 8px; }
      .news-card { border: 1px solid var(--border); padding: 12px 14px; border-radius: 12px; background: #fff; }
      .news-title { font-weight: 600; font-size: 14px; margin: 2px 0; }
      .news-meta { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
      .news-desc { font-size: 13px; color: #374151; }
      .divider { height: 1px; background: var(--border); margin: 28px 0 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ HEADER ------------------------------
st.markdown('<div class="app-container">', unsafe_allow_html=True)
st.markdown('<div class="title">Finance Projects — Investment Banking Track</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">M&A monitoring, DCF valuation, and systematic backtesting — professional tools for deal-driven workflows.</div>', unsafe_allow_html=True)

# ------------------------------ GRID CARDS ------------------------------
st.markdown('<div class="grid">', unsafe_allow_html=True)

# Card 1: M&A Dashboard
st.markdown(
    """
    <div class="card">
      <div class="card-title">M&A Deals Dashboard</div>
      <div class="card-desc">FMP feed, time filters, sector/region/value screening, interactive timeline and table.</div>
      <div class="btn-row">
        <a class="btn btn-link" href="pages/1_M&A_Deals_Dashboard.py">Open</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Card 2: DCF Lab
st.markdown(
    """
    <div class="card">
      <div class="card-title">DCF Valuation Lab</div>
      <div class="card-desc">FCF projection, WACC, terminal value, scenario sensitivity, and PDF export.</div>
      <div class="btn-row">
        <a class="btn btn-link" href="pages/2_DCF_Lab.py">Open</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Card 3: Backtester
st.markdown(
    """
    <div class="card">
      <div class="card-title">Equity & Bond Backtester</div>
      <div class="card-desc">Historical returns, drawdowns, Sharpe, CAGR, benchmark comparison with adjustable parameters.</div>
      <div class="btn-row">
        <a class="btn btn-link" href="pages/3_Backtester_Equity_Bonds.py">Open</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)  # end grid

# ------------------------------ NEWS PANEL ------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Global News</div>', unsafe_allow_html=True)

# Barre de recherche (simple) + affichage résultats
q_col1, q_col2 = st.columns([2, 1])
with q_col1:
    query = st.text_input("Search topic or company (e.g., Goldman Sachs, M&A, LBO):", value="M&A")
with q_col2:
    page_size = st.number_input("Results", min_value=3, max_value=20, value=8, step=1)

articles = fetch_news(query=query, page_size=page_size) if query else []

if not articles:
    st.caption("No news available.")
else:
    grid_cols = st.columns(2)
    for idx, a in enumerate(articles):
        col = grid_cols[idx % 2]
        with col:
            with st.container():
                st.markdown('<div class="news-card">', unsafe_allow_html=True)
                title = a.get("title", "")
                source = (a.get("source") or {}).get("name", "")
                published = a.get("publishedAt", "")
                desc = a.get("description", "") or ""
                url = a.get("url", "")

                st.markdown(f'<div class="news-title">{title}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="news-meta">{source} · {published}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="news-desc">{desc}</div>', unsafe_allow_html=True)
                if url:
                    st.link_button("Open article", url)
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end app-container
