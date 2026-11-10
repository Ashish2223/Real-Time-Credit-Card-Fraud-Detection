"""Model wrapper used by the API: loading, scoring, risk bands and SHAP reasons."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from .. import explain
from ..features import engineer_features


class FraudModel:
    def __init__(self, models_dir: str | Path):
        models_dir = Path(models_dir)
        self.meta = json.loads((models_dir / "metadata.json").read_text())
        self.name: str = self.meta["serving_model"]
        self.threshold: float = float(self.meta["threshold"])
        self.model = joblib.load(models_dir / f"{self.name}.joblib")
        self._explainer = None

    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "high"
        return "medium" if p >= 0.5 * self.threshold else "low"

    def predict(self, records: list[dict], explain_top_k: int = 0) -> list[dict]:
        t0 = time.perf_counter()
        df = pd.DataFrame(records)
        X = engineer_features(df)
        probs = self.model.predict_proba(X)[:, 1]

        factors = None
        if explain_top_k:
            if self._explainer is None:
                self._explainer = explain.make_explainer(self.model)
            factors = explain.top_contributions(self._explainer, X, explain_top_k)

        per_row_ms = (time.perf_counter() - t0) * 1000 / len(records)
        out = []
        for i, (rec, p) in enumerate(zip(records, probs)):
            p = float(p)
            out.append(
                {
                    "transaction_id": rec.get("transaction_id"),
                    "fraud_probability": round(p, 6),
                    "is_fraud": p >= self.threshold,
                    "risk_level": self.risk_level(p),
                    "threshold": round(self.threshold, 6),
                    "model": self.name,
                    "latency_ms": round(per_row_ms, 3),
                    "top_factors": factors[i] if factors else None,
                }
            )
        return out
