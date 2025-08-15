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
        .app-container { max-width: 1100px; margin: auto; padding: 20px; }
        .page-title { font-size: 28px; font-weight: bold; }
        .page-subtitle { font-size: 14px; color: #6b7280; margin-bottom: 20px; }
        .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: white; margin-bottom: 16px; }
        .card-title { font-size: 18px; font-weight: bold; }
        .card-desc { font-size: 13px; color: #6b7280; }
        .news-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: white; margin-bottom: 10px; }
        .news-title { font-weight: bold; }
        .news-meta { font-size: 12px; color: gray; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container
st.markdown('<div class="app-container">', unsafe_allow_html=True)

st.markdown('<div class="page-title">Finance Projects</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Tools for Investment Banking: M&A Dashboard, DCF Lab, Portfolio Backtester.</div>',
    unsafe_allow_html=True
)

# Navigation cards
projects = [
    ("1_M&A_Deals_Dashboard.py", "M&A Deals Dashboard", "Live global deal feed."),
    ("2_DCF_Lab.py", "DCF Lab", "Interactive discounted cash flow valuation."),
    ("3_Backtester_Equity_Bonds.py", "Equity/Bond Backtester", "Portfolio analysis with key metrics."),
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

# News section
st.markdown("### Investment Banking News")
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
    st.info("No news available.")

if st.button("Reload news"):
    st.session_state.news_page += 1
    st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)  # end container
