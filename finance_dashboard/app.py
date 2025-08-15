import streamlit as st
import os
from utils.ga import inject_ga4

# GA4 Tracking
st.set_page_config(page_title="Finance Projects — IB Portfolio", layout="wide")
inject_ga4()

# CSS for premium banker look
st.markdown("""
<style>
    body { background-color: #f8fafc; }
    .project-section {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 40px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .project-section:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .project-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 15px;
        color: #0f172a;
    }
    .project-desc {
        font-size: 15px;
        color: #444;
        text-align: justify;
        line-height: 1.6;
        margin-bottom: 20px;
        min-height: 90px;
    }
    .btn {
        display: inline-block;
        padding: 10px 24px;
        background-color: #0f172a;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-size: 14px;
        transition: background-color 0.2s ease;
    }
    .btn:hover {
        background-color: #1e293b;
    }
    video, img {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- [Banking Market Intelligence](#banking-market-intelligence-dashboard)
- [M&A Deals Dashboard](#ma-deals-dashboard)
- [DCF Lab](#dcf-lab)
- [Equity/Bond Backtester](#equitybond-backtester)
---
- [News](#news)
""")

# Main content
st.markdown("## Finance Projects")
st.write("A selection of advanced finance tools designed for investment banking workflows, with intuitive interfaces and real-time insights.")

# Media folder
media_folder = "media"

# List of projects
projects = [
    {
        "id": "banking-market-intelligence-dashboard",
        "title": "Banking Market Intelligence Dashboard",
        "desc": "An integrated platform providing live macroeconomic data, sector analysis, and market intelligence, tailored for investment banking professionals. Features real-time charts, comparative analytics, and curated insights for strategic decision-making.",
        "media": "banking_demo.mp4",
        "page": "pages/0_Legacy_Analytics.py"
    },
    {
        "id": "ma-deals-dashboard",
        "title": "M&A Deals Dashboard",
        "desc": "Tracks global mergers and acquisitions with live updates, filtering by sector, region, and deal size. Offers dynamic visualizations, deal timelines, and integrated financial metrics to support pitchbook preparation and client advisory.",
        "media": "ma_demo.mp4",
        "page": "pages/1_M&A_Deals_Dashboard.py"
    },
    {
        "id": "dcf-lab",
        "title": "DCF Lab",
        "desc": "An interactive discounted cash flow modeling environment where assumptions such as WACC, growth rates, and exit multiples can be adjusted on the fly. Outputs instant valuations with sensitivity analysis for deal pricing and fairness opinions.",
        "media": "dcf_demo.mp4",
        "page": "pages/2_DCF_Lab.py"
    },
    {
        "id": "equitybond-backtester",
        "title": "Equity/Bond Backtester",
        "desc": "Portfolio simulation tool allowing backtesting of equities and fixed income assets. Provides key performance indicators including CAGR, Sharpe ratio, drawdowns, and volatility metrics, with customizable date ranges and allocation weights.",
        "media": "backtester_demo.mp4",
        "page": "pages/3_Backtester_Equity_Bonds.py"
    }
]

# Render projects
for p in projects:
    st.markdown(f'<div class="project-section" id="{p["id"]}">', unsafe_allow_html=True)

    # Build full media path
    media_path = os.path.join(media_folder, p.get("media", ""))
    if media_path and os.path.exists(media_path):
        if media_path.endswith(".mp4"):
            st.video(media_path)
        else:
            st.image(media_path)
    else:
        st.write(f"Media file not found: {media_path}")

    st.markdown(f'<div class="project-title">{p["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="project-desc">{p["desc"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<a href="{p["page"]}" class="btn">Open Project</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# News link at bottom
st.markdown("""
<div class="project-section" id="news">
    <div class="project-title">Finance News</div>
    <div class="project-desc">Stay updated with the latest financial headlines, market movements, and industry insights from trusted sources.</div>
    <a href="pages/4_news.py" class="btn">View News</a>
</div>
""", unsafe_allow_html=True)
