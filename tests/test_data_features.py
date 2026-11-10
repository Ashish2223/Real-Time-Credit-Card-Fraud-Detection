import numpy as np

from fraud_detection.config import FEATURE_COLS, RAW_COLS, TARGET
from fraud_detection.data import generate_synthetic
from fraud_detection.features import engineer_features


def test_synthetic_schema_and_imbalance():
    df = generate_synthetic(20_000, fraud_rate=0.002, seed=1)
    assert list(df.columns) == [*RAW_COLS, TARGET]
    assert len(df) == 20_000
    assert 0.001 < df[TARGET].mean() < 0.01
    assert (df["Amount"] >= 0).all()


def test_synthetic_is_deterministic():
    a, b = generate_synthetic(2_000, seed=3), generate_synthetic(2_000, seed=3)
    assert a.equals(b)


def test_engineer_features():
    df = generate_synthetic(1_000, seed=2)
    X = engineer_features(df)
    assert list(X.columns) == FEATURE_COLS
    assert X["hour"].between(0, 23).all()
    assert np.isfinite(X.to_numpy()).all()
