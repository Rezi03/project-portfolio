CREATE TABLE IF NOT EXISTS deals (
  id TEXT PRIMARY KEY,
  announced_date TEXT,
  acquirer TEXT,
  target TEXT,
  value_usd REAL,
  status TEXT,
  source_url TEXT,
  filing_url TEXT,
  sector TEXT,
  region TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);