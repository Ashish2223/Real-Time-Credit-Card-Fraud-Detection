"""Feature engineering shared by training and the serving API (prevents train/serve skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, V_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns (Time, V1..V28, Amount) -> model features.

    * ``log_amount``: log1p tames the heavy right tail of transaction amounts.
    * ``hour``: hour of day derived from the seconds-since-start ``Time`` column.
    """
    out = df[V_COLS].astype(float).copy()
    out["log_amount"] = np.log1p(df["Amount"].astype(float).clip(lower=0))
    out["hour"] = ((df["Time"].astype(float) // 3600) % 24).astype(float)
    return out[FEATURE_COLS]
