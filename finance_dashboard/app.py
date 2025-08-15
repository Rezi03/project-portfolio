import streamlit as st
from utils.ga import inject_ga4
from utils.news import fetch_news_page

# Page configuration
st.set_page_config(
    page_title="Finance Projects — Investment Banking Track",
    layout="wide",
)

# Inject GA4 analytics
inject_ga4()

# Global minimal style
st.markdown(
    """
    <style>
        body { background-color: #f8fafc; }
        .app-container { max-width: 1100px; margin: auto; padding: 10px 20px; }
        .page-title { font-size: 28px; font-weight: bold; margin-bottom: 4px; }
        .page-subtitle { font-size: 14px; color: #6b7280; margin-bottom: 24px; }
        .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; background: white; margin-bottom: 12px; }
        .card-title { font-size: 18px; font-weight: bold; }
        .card-desc { font-size: 13px; color: #6b7280; }
        .news-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 14px; background: white; margin-bottom: 12px; }
        .news-title { font-weight: bold; }
        .news-meta { font-size: 12px; color: gray; }
        hr.section-divider { margin: 32px 0 20px 0; border: none; border-top: 1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Header
st.markdown('<div class="page-title">Finance Projects</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Specialized tools for investment banking professionals and candidates.</div>',
    unsafe_allow_html=True
)

# Project cards
projects = [
    ("Banking_Market_Intelligence.py", "Banking Market Intelligence Dashboard", "Aggregated market data, macro indicators, and sector trends in one interactive dashboard."),
    ("1_M&A_Deals_Dashboard.py", "M&A Deals Dashboard", "Live feed of global mergers & acquisitions activity with key deal metrics."),
    ("2_DCF_Lab.py", "DCF Lab", "Interactive discounted cash flow valuation model with custom parameters."),
    ("3_Backtester_Equity_Bonds.py", "Equity/Bond Backtester", "Portfolio backtesting with CAGR, Sharpe ratio, and drawdown analytics."),
]

for file, title, desc in projects:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-desc">{desc}</div>', unsafe_allow_html=True)
    try:
        st.page_link(f"pages/{file}", label=f"Open {title}")
    except:
        st.write("Use the sidebar to navigate.")
    st.markdown('</div>', unsafe_allow_html=True)

# Divider before news
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# News section
st.markdown("### Investment Banking News")
st.caption("Curated news feed with limited pagination for relevance and quality.")

if "news_page" not in st.session_state:
    st.session_state.news_page = 1

articles = fetch_news_page(page=st.session_state.news_page)

if articles:
    for a in articles:
        st.markdown('<div class="news-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="news-title">{a.get("title", "")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="news-meta">{a.get("source", {}).get("name", "")} — {a.get("publishedAt", "")[:10]}</div>', unsafe_allow_html=True)
        st.write(a.get("description", ""))
        if a.get("url"):
            st.link_button("Open article", a["url"])
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No news available at the moment.")

if st.button("Reload news"):
    st.session_state.news_page += 1
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # end container
