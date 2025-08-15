import os
import requests
from typing import List, Dict, Any, Optional

_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news(
    query: str,
    api_key: Optional[str] = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 8,
) -> List[Dict[str, Any]]:
    # Récupération de la clé API (priorité aux secrets Streamlit si présents)
    try:
        import streamlit as st
        key = api_key or st.secrets.get("NEWSAPI_KEY", "")
    except Exception:
        key = api_key or os.getenv("NEWSAPI_KEY", "")

    if not key:
        return []

    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size,
        "apiKey": key,
    }
    r = requests.get(_ENDPOINT, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", []) or []
