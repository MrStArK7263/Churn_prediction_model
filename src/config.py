"""
config.py – central place for all paths and constants.
Change MODEL_VERSION here when you retrain, everything else picks it up automatically.
"""

import os
from pathlib import Path

# ── Project root (two levels up from this file: src/ → churn-ml-project/) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data_23.csv"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = "1.0.0"
MODEL_FILENAME = f"churn_model_v{MODEL_VERSION}.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ── Training ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42          # fixed seed for reproducibility everywhere
TEST_SIZE = 0.20           # 80/20 split
CV_FOLDS = 5               # cross-validation folds

# ── Target column ─────────────────────────────────────────────────────────────
TARGET_COL = "Churn"
POSITIVE_CLASS = "Yes"     # "Yes" means the customer churned

# ── Columns to drop before modelling ──────────────────────────────────────────
# customerID is just an identifier; it has no predictive signal
ID_COL = "customerID"
COLS_TO_DROP = [ID_COL]

# ── Numeric columns (raw, before feature engineering) ─────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ── Categorical columns (raw, before feature engineering) ─────────────────────
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# ── Engineered feature names (created in feature_engineering.py) ───────────────
ENGINEERED_NUMERIC = ["avg_monthly_spend", "num_services", "tenure_x_monthly"]
ENGINEERED_BINARY  = ["has_internet", "has_phone", "auto_pay", "is_monthly_fiber"]
TENURE_GROUP_COL   = "tenure_group"

# ── Decision threshold (can be overridden by model_metadata.json) ─────────────
DEFAULT_THRESHOLD = 0.50
