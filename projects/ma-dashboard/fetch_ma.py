import requests
import json
import os
from datetime import datetime

# Essayer de charger la clé depuis secrets.toml (local) sinon depuis variable d'environnement
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    FMP_API_KEY = secrets.get("FMP_API_KEY", "")
except Exception:
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "deals.json")

def fetch_ma_deals():
    if not FMP_API_KEY:
        raise ValueError("❌ Missing FMP_API_KEY")
    url = f"https://financialmodelingprep.com/api/v4/mergers-acquisitions?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}")
    return response.json()

if __name__ == "__main__":
    print(f"[{datetime.utcnow()}] Fetching M&A deals from FMP...")
    deals = fetch_ma_deals()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deals, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(deals)} deals to {OUTPUT_FILE}")
