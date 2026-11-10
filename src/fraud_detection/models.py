"""Model zoo: Random Forest, XGBoost (supervised) and Isolation Forest (unsupervised)."""
from __future__ import annotations

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def build_random_forest(fast: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=60 if fast else 250,
        max_depth=16,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",  # re-weights the rare class inside every bootstrap
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_xgboost(scale_pos_weight: float, fast: bool = False) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=150 if fast else 600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,  # n_legit / n_fraud
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_isolation_forest(contamination: float, fast: bool = False) -> IsolationForest:
    # Unsupervised: never sees labels. ``contamination`` is just the expected anomaly share.
    return IsolationForest(
        n_estimators=100 if fast else 300,
        contamination=min(max(contamination, 1e-4), 0.5),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
