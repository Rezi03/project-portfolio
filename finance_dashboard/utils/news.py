from typing import List, Dict, Optional
import requests
import streamlit as st

_ENDPOINT = "https://newsapi.org/v2/everything"
_DEFAULT_QUERY = (
    "investment banking OR M&A OR mergers acquisitions OR leveraged buyout OR IPO"
)

def fetch_news_page(
    page: int = 1,
    page_size: int = 10,
    query: str = _DEFAULT_QUERY,
    language: str = "en",
    sort_by: str = "publishedAt",
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch a single page of curated Investment Banking news.
    No free-text search is allowed, only pagination.
    """
    key = api_key or st.secrets.get("NEWSAPI_KEY", "")
    if not key:
        return []

    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size,
        "page": page,
        "apiKey": key,
    }
    try:
        r = requests.get(_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []
