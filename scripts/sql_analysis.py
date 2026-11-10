"""Load the dataset into SQLite and run the exploratory SQL in sql/eda_queries.sql.

    python scripts/sql_analysis.py --load [--data data/creditcard.csv]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from fraud_detection.config import DATA_DIR, ROOT
from fraud_detection.data import load_data

SQL_DIR = ROOT / "sql"


def load(db_path: Path, data_path: str | None) -> None:
    df, source = load_data(data_path)
    with closing(sqlite3.connect(db_path)) as con:
        con.executescript("DROP TABLE IF EXISTS transactions;")
        con.executescript((SQL_DIR / "schema.sql").read_text())
        df.to_sql("transactions", con, if_exists="append", index=False)
        con.commit()
    print(f"Loaded {len(df):,} rows ({source}) into {db_path}")


def run_queries(db_path: Path) -> None:
    text = (SQL_DIR / "eda_queries.sql").read_text()
    blocks = re.split(r"^-- name: (\w+)\s*$", text, flags=re.MULTILINE)[1:]
    with closing(sqlite3.connect(db_path)) as con:
        for name, query in zip(blocks[::2], blocks[1::2]):
            print(f"\n=== {name} ===")
            print(pd.read_sql_query(query, con).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="(Re)load the transactions table first")
    ap.add_argument("--data", default=None)
    ap.add_argument("--db", default=str(DATA_DIR / "fraud.db"))
    a = ap.parse_args()
    db = Path(a.db)
    if a.load or not db.exists():
        load(db, a.data)
    run_queries(db)
