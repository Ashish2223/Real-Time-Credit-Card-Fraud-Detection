-- Analytical table (loaded from creditcard.csv by scripts/sql_analysis.py)
CREATE TABLE IF NOT EXISTS transactions (
    Time   REAL,      -- seconds since first transaction
    V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL,
    V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL,
    V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL,
    Amount REAL,
    Class  INTEGER CHECK (Class IN (0, 1))  -- 1 = fraud
);
CREATE INDEX IF NOT EXISTS idx_tx_class ON transactions (Class);
CREATE INDEX IF NOT EXISTS idx_tx_time  ON transactions (Time);

-- Online scoring audit log (written by the FastAPI service)
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    TEXT,
    scored_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    amount            REAL,
    fraud_probability REAL NOT NULL,
    is_fraud          INTEGER NOT NULL,
    risk_level        TEXT NOT NULL,
    threshold         REAL NOT NULL,
    model             TEXT NOT NULL,
    latency_ms        REAL
);
