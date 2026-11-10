import pytest

from fraud_detection.data import generate_synthetic
from fraud_detection.train import train_pipeline


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic(8_000, fraud_rate=0.02, seed=7)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, small_df):
    out = tmp_path_factory.mktemp("artifacts")
    train_pipeline(small_df, out / "models", out / "reports", fast=True, save_plots=False, source="synthetic")
    return out / "models"
