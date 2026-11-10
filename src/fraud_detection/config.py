"""Central configuration: paths, column names and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"  # Kaggle "Credit Card Fraud Detection" (ULB) dataset

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", *V_COLS, "Amount"]
TARGET = "Class"
FEATURE_COLS = [*V_COLS, "log_amount", "hour"]

RANDOM_STATE = 42
