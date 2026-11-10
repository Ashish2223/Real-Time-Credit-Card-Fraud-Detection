"""Tiny SQLite audit log of every scored transaction (swap for Postgres in production)."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

DDL = """
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
CREATE INDEX IF NOT EXISTS idx_predictions_scored_at ON predictions (scored_at);
"""


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as con:
        con.executescript(DDL)
        con.commit()


def log_predictions(path: str | Path, rows: list[dict]) -> None:
    with closing(sqlite3.connect(path)) as con:
        con.executemany(
            "INSERT INTO predictions (transaction_id, amount, fraud_probability, is_fraud, risk_level, threshold, model, latency_ms) "
            "VALUES (:transaction_id, :amount, :fraud_probability, :is_fraud, :risk_level, :threshold, :model, :latency_ms)",
            rows,
        )
        con.commit()


def stats(path: str | Path) -> dict:
    with closing(sqlite3.connect(path)) as con:
        total, flagged, avg_p, avg_lat = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(is_fraud),0), COALESCE(AVG(fraud_probability),0), COALESCE(AVG(latency_ms),0) FROM predictions"
        ).fetchone()
    return {
        "total_scored": total,
        "flagged_as_fraud": int(flagged),
        "flag_rate": round(flagged / total, 6) if total else 0.0,
        "avg_fraud_probability": round(avg_p, 6),
        "avg_latency_ms": round(avg_lat, 3),
    }
