import numpy as np
import pytest

from fraud_detection.evaluate import best_f1_threshold, evaluate


def test_perfect_separation():
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)])
    t = best_f1_threshold(y, s)
    m = evaluate(y, s, t)
    assert m["f1"] == 1.0 and m["pr_auc"] == pytest.approx(1.0) and m["fp"] == 0 and m["fn"] == 0


def test_no_predictions_is_safe():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.3, 0.4]), threshold=2.0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
