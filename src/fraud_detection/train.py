"""Training pipeline: split -> fit 3 models -> tune threshold on validation -> evaluate on test.

Run:  python -m fraud_detection.train [--data data/creditcard.csv] [--fast] [--no-plots]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import train_test_split

from . import explain
from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from .data import load_data
from .evaluate import best_f1_threshold, evaluate
from .features import engineer_features
from .models import build_isolation_forest, build_random_forest, build_xgboost

log = logging.getLogger("fraud.train")
DISPLAY = {"random_forest": "Random Forest", "xgboost": "XGBoost", "isolation_forest": "Isolation Forest"}


def _scores(name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Higher score = more likely fraud, for every model family."""
    if name == "isolation_forest":
        return -model.score_samples(X)  # sklearn: lower score_samples = more anomalous
    return model.predict_proba(X)[:, 1]


def train_pipeline(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    fast: bool = False,
    save_plots: bool = True,
    source: str = "unknown",
) -> dict:
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y = engineer_features(df), df[TARGET].astype(int)

    # Stratified splits keep the fraud ratio identical in train / val / test.
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=RANDOM_STATE)
    n_pos, n_neg = int(y_fit.sum()), int((y_fit == 0).sum())
    log.info("fit=%d val=%d test=%d | fraud rate=%.4f%%", len(X_fit), len(X_val), len(X_test), 100 * y.mean())

    models = {
        "random_forest": build_random_forest(fast),
        "isolation_forest": build_isolation_forest(n_pos / len(y_fit), fast),
    }

    # XGBoost: pick the class weight on VALIDATION PR-AUC. The full n_neg/n_pos weight (~580x on the
    # real data) can make boosting stop after a handful of trees; milder weights often generalise better.
    full_w = n_neg / max(n_pos, 1)
    weight_search, best_xgb = [], None
    t_xgb = time.perf_counter()
    for label, w in (("balanced (n_neg/n_pos)", full_w), ("sqrt-balanced", full_w**0.5), ("none (1.0)", 1.0)):
        cand = build_xgboost(w, fast)
        cand.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        val_ap = float(average_precision_score(y_val, cand.predict_proba(X_val)[:, 1]))
        weight_search.append({"scale_pos_weight": label, "value": round(w, 2), "val_pr_auc": round(val_ap, 4), "trees": int(cand.best_iteration) + 1})
        if best_xgb is None or val_ap > best_xgb[0]:
            best_xgb = (val_ap, cand, w)
    xgb_seconds = time.perf_counter() - t_xgb
    models["xgboost"] = best_xgb[1]
    models = {k: models[k] for k in ("random_forest", "xgboost", "isolation_forest")}  # stable display order
    pd.DataFrame(weight_search).to_csv(reports_dir / "xgb_weight_search.csv", index=False)
    log.info("XGBoost weight search (validation): %s", weight_search)

    results, test_scores, timings = {}, {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        if name == "xgboost":
            pass  # already fitted by the search above
        elif name == "isolation_forest":
            model.fit(X_fit)  # labels are never used
        else:
            model.fit(X_fit, y_fit)
        timings[name] = xgb_seconds if name == "xgboost" else time.perf_counter() - t0

        threshold = best_f1_threshold(y_val, _scores(name, model, X_val))  # tuned on VALIDATION only
        test_scores[name] = _scores(name, model, X_test)
        results[name] = {**evaluate(y_test, test_scores[name], threshold), "train_seconds": round(timings[name], 2)}
        joblib.dump(model, models_dir / f"{name}.joblib")
        log.info("%-16s PR-AUC=%.4f  recall=%.3f  precision=%.3f", name, results[name]["pr_auc"], results[name]["recall"], results[name]["precision"])

    # Serve the best *supervised* model (they return calibrated-ish probabilities for the API).
    serving = max(("random_forest", "xgboost"), key=lambda n: results[n]["pr_auc"])
    metadata = {
        "serving_model": serving,
        "threshold": results[serving]["threshold"],
        "feature_cols": list(X.columns),
        "metrics": results,
        "xgboost_weight_search": weight_search,
        "data_source": source,
        "n_rows": len(df),
        "fraud_rate": float(y.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    table = (
        pd.DataFrame(results).T.rename(index=DISPLAY)[["pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_1pct_fpr", "tp", "fp", "fn", "train_seconds"]]
        .round(4)
    )
    table.to_csv(reports_dir / "model_comparison.csv")
    (reports_dir / "model_comparison.md").write_text(table.to_markdown() if _has_tabulate() else table.to_string())

    if save_plots:
        _save_plots(models, results, test_scores, y_test, X_test, reports_dir / "figures", serving)
    return metadata


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _save_plots(models, results, test_scores, y_test, X_test, fig_dir: Path, serving: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in test_scores.items():
        PrecisionRecallDisplay.from_predictions(y_test, s, name=DISPLAY[name], ax=ax)
    ax.axhline(y_test.mean(), ls="--", c="grey", label=f"No-skill ({y_test.mean():.4f})")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, s) in zip(axes, test_scores.items()):
        pred = (s >= results[name]["threshold"]).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Legit", "Fraud"], cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(DISPLAY[name])
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    try:  # SHAP is best-effort: never let a plotting problem break training
        sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
        explain.save_summary_plots(models[serving], sample, fig_dir)
    except Exception as exc:  # pragma: no cover
        log.warning("SHAP plots skipped: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="Path to creditcard.csv (default: data/creditcard.csv, else synthetic)")
    ap.add_argument("--synthetic-rows", type=int, default=100_000)
    ap.add_argument("--fast", action="store_true", help="Smaller models for quick runs / CI")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    df, source = load_data(args.data, args.synthetic_rows)
    meta = train_pipeline(df, fast=args.fast, save_plots=not args.no_plots, source=source)
    print(f"\nServing model: {meta['serving_model']}  (threshold={meta['threshold']:.4f}, data={source})")
    print((REPORTS_DIR / "model_comparison.md").read_text())


if __name__ == "__main__":
    main()
