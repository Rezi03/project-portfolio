import streamlit as st
import streamlit.components.v1 as components

_HTML_TEMPLATE = """
<script async src="https://www.googletagmanager.com/gtag/js?id={mid}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag() {{ dataLayer.push(arguments); }}
  gtag('js', new Date());
  gtag('config', '{mid}');
</script>
"""

def inject_ga4(measurement_id: str | None = None) -> None:
    """
    Inject the Google Analytics 4 tracking script.
    Reads GA4_MEASUREMENT_ID from Streamlit secrets if not provided.
    """
    mid = measurement_id or st.secrets.get("GA4_MEASUREMENT_ID", "")
    if not mid:
        return
    components.html(_HTML_TEMPLATE.format(mid=mid), height=0, scrolling=False)