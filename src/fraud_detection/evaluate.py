"""Metrics for imbalanced classification. Accuracy is useless at 0.17% fraud - use PR-AUC & co."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on the precision-recall curve (pick it on validation data!)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Share of fraud caught while flagging at most ``max_fpr`` of legitimate transactions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
