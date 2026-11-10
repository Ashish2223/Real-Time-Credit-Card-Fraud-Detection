"""SHAP explainability for tree models (global summary plots + per-transaction reasons)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def make_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _positive_class(values) -> np.ndarray:
    """Normalise SHAP output across versions/models to (n_samples, n_features) for the fraud class."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    return _positive_class(explainer.shap_values(X))


def top_contributions(explainer: shap.TreeExplainer, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
    """Top-k features pushing each row towards (+) or away from (-) 'fraud'."""
    sv = shap_values(explainer, X)
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(sv[i]))[:k]
        out.append(
            [
                {
                    "feature": X.columns[j],
                    "value": float(X.iloc[i, j]),
                    "shap_value": float(sv[i, j]),
                    "direction": "increases_risk" if sv[i, j] > 0 else "decreases_risk",
                }
                for j in order
            ]
        )
    return out


def save_summary_plots(model, X_sample: pd.DataFrame, out_dir: Path, prefix: str = "shap") -> list[Path]:
    """Beeswarm + bar plots of global feature importance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sv = shap_values(make_explainer(model), X_sample)
    paths = []
    for kind, name in (("dot", "summary"), ("bar", "importance")):
        plt.figure()
        shap.summary_plot(sv, X_sample, plot_type=kind, max_display=15, show=False)
        plt.tight_layout()
        p = out_dir / f"{prefix}_{name}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close()
        paths.append(p)
    return paths
