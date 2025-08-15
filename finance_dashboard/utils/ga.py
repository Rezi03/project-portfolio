# utils/ga.py
import streamlit as st
import streamlit.components.v1 as components

_HTML = """
<!-- GA4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id={mid}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{mid}');
</script>
"""

def inject_ga4(measurement_id: str | None = None) -> None:
    mid = measurement_id or st.secrets.get("GA4_MEASUREMENT_ID", "")
    if not mid:
        return
    components.html(_HTML.format(mid=mid), height=0, scrolling=False)