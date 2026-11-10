"""Data loading + a realistic synthetic generator.

The real benchmark is the ULB/Kaggle credit-card dataset (284,807 rows, 0.172% fraud, columns
Time, V1..V28 [PCA components], Amount, Class). If ``data/creditcard.csv`` is missing, a synthetic
dataset with the *same schema* and a similar imbalance is generated so the whole pipeline runs
out-of-the-box (CI, Docker, quick demos).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_COLS, RAW_DATA_PATH, TARGET, V_COLS

log = logging.getLogger(__name__)

# Features whose distribution shifts for fraudulent transactions (mimics the real dataset,
# where V14, V12, V10, V17, V4, V11 ... are the most discriminative components).
_FRAUD_SHIFTS = {
    "V1": -2.0, "V2": 1.5, "V3": -2.5, "V4": 2.0, "V5": -1.0, "V7": -1.5, "V9": -1.5,
    "V10": -2.5, "V11": 2.0, "V12": -3.0, "V14": -3.5, "V16": -1.5, "V17": -2.0, "V18": -1.0,
}


def generate_synthetic(n_samples: int = 100_000, fraud_rate: float = 0.002, seed: int = 42) -> pd.DataFrame:
    """Create a highly imbalanced, partially-overlapping synthetic transaction dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_samples * fraud_rate)), 10)
    n_legit = n_samples - n_fraud
    scales = np.linspace(1.9, 0.4, len(V_COLS))

    legit = rng.normal(0.0, scales, size=(n_legit, len(V_COLS)))
    fraud = rng.normal(0.0, scales * 1.25, size=(n_fraud, len(V_COLS)))

    shift = np.array([_FRAUD_SHIFTS.get(c, 0.0) for c in V_COLS])
    # 25% of frauds are "stealthy": only a weak version of the fraud signature.
    stealth = rng.random(n_fraud) < 0.25
    fraud += shift * np.where(stealth, 0.35, 1.0)[:, None]

    # Amounts: legit ~ log-normal; fraud = card-testing micro-payments or large cash-outs.
    legit_amt = rng.lognormal(mean=3.4, sigma=1.3, size=n_legit)
    kind = rng.random(n_fraud)
    fraud_amt = np.where(
        kind < 0.4,
        rng.uniform(0.5, 5.0, n_fraud),
        np.where(kind < 0.7, rng.lognormal(3.4, 1.3, n_fraud), rng.lognormal(5.5, 0.8, n_fraud)),
    )

    # Time: 2 days of seconds. Legit follows a daily cycle; fraud skews toward the night.
    hours = np.arange(24)
    p_legit = 1.2 + np.sin((hours - 6) / 24 * 2 * np.pi)
    p_fraud = np.where(hours < 6, 3.0, 1.0)
    p_legit, p_fraud = p_legit / p_legit.sum(), p_fraud / p_fraud.sum()

    def _times(n: int, p: np.ndarray) -> np.ndarray:
        day = rng.integers(0, 2, n)
        hour = rng.choice(hours, n, p=p)
        return day * 86_400 + hour * 3_600 + rng.uniform(0, 3_600, n)

    df = pd.concat(
        [
            pd.DataFrame(legit, columns=V_COLS).assign(Amount=legit_amt, Time=_times(n_legit, p_legit), Class=0),
            pd.DataFrame(fraud, columns=V_COLS).assign(Amount=fraud_amt, Time=_times(n_fraud, p_fraud), Class=1),
        ],
        ignore_index=True,
    )
    df["Amount"] = df["Amount"].round(2)
    df = df.sort_values("Time").reset_index(drop=True)
    return df[[*RAW_COLS, TARGET]]


def load_data(path: str | Path | None = None, synthetic_rows: int = 100_000) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source)`` where source is 'kaggle' or 'synthetic'."""
    path = Path(path) if path else RAW_DATA_PATH
    if path.exists():
        log.info("Loading %s", path)
        df = pd.read_csv(path)
        missing = set([*RAW_COLS, TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return df[[*RAW_COLS, TARGET]], "kaggle"
    log.warning("%s not found - generating synthetic data (%d rows).", path, synthetic_rows)
    return generate_synthetic(synthetic_rows), "synthetic"
