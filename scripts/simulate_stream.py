"""Replay transactions against the running API to demo real-time scoring.

    uvicorn fraud_detection.api.main:app &      # in another terminal
    python scripts/simulate_stream.py --n 50 --delay 0.2
"""
from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from fraud_detection.config import RAW_COLS, TARGET
from fraud_detection.data import load_data

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=50, help="Number of transactions to stream")
ap.add_argument("--fraud-share", type=float, default=0.2, help="Share of fraud in the demo stream (boosted for visibility)")
ap.add_argument("--delay", type=float, default=0.1)
args = ap.parse_args()

df, _ = load_data()
n_fraud = int(args.n * args.fraud_share)
sample = pd.concat(
    [
        df[df[TARGET] == 1].sample(n_fraud, random_state=1),
        df[df[TARGET] == 0].sample(args.n - n_fraud, random_state=1),
    ]
).sample(frac=1, random_state=1)

caught = missed = false_alarms = 0
with httpx.Client(base_url=args.url, timeout=10) as client:
    for i, (_, row) in enumerate(sample.iterrows()):
        payload = {"transaction_id": f"sim-{i:04d}", **{c: float(row[c]) for c in RAW_COLS}}
        r = client.post("/predict", json=payload).json()
        actual = int(row[TARGET])
        caught += actual and r["is_fraud"]
        missed += actual and not r["is_fraud"]
        false_alarms += (not actual) and r["is_fraud"]
        flag = "FRAUD " if r["is_fraud"] else "ok    "
        print(f"{r['transaction_id']}  {flag} p={r['fraud_probability']:.3f}  risk={r['risk_level']:<6} amount={row['Amount']:>8.2f}  actual={'fraud' if actual else 'legit'}  {r['latency_ms']:.1f}ms")
        time.sleep(args.delay)

print(f"\ncaught={caught} missed={missed} false_alarms={false_alarms}")
print(httpx.get(f"{args.url}/stats").json())
