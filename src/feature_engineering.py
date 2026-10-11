"""
feature_engineering.py – reusable feature-creation functions.

These functions are called from:
  - The Jupyter notebook (for EDA + training)
  - train.py (to build the sklearn Pipeline)
  - backend/app/predictor.py (so the API makes identical features)

Keeping all feature logic here ensures training and serving never diverge.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: list of add-on service columns
# ─────────────────────────────────────────────────────────────────────────────
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


# ─────────────────────────────────────────────────────────────────────────────
# PURE-PANDAS FEATURE BUILDER (used in the notebook for exploration)
# ─────────────────────────────────────────────────────────────────────────────

def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a copy of the DataFrame.

    Features created:
      1. tenure_group        – bins tenure into 4 bands (easy for beginners to read)
      2. avg_monthly_spend   – TotalCharges / (tenure + 1)
                               WHY: avoids divide-by-zero for tenure=0 customers
      3. num_services        – count of subscribed add-on services
                               WHY: customers with more services are more invested → less likely to churn
      4. has_internet        – 1 if InternetService != 'No'
      5. has_phone           – 1 if PhoneService == 'Yes'
      6. auto_pay            – 1 if PaymentMethod contains 'automatic'
                               WHY: customers on auto-pay are more stable
      7. is_monthly_fiber    – interaction: month-to-month contract AND fiber optic
                               WHY: this combination is the highest-churn segment
      8. tenure_x_monthly    – tenure × MonthlyCharges
                               WHY: captures whether a long-tenure, high-charge customer is at risk
    """
    df = df.copy()

    # ── 1. tenure_group ───────────────────────────────────────────────────────
    # Convert tenure to plain int for pd.cut (Int64 nullable int works fine too)
    tenure_numeric = pd.to_numeric(df["tenure"], errors="coerce")
    df["tenure_group"] = pd.cut(
        tenure_numeric,
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        right=True,
        include_lowest=True,
    )
    logger.info("Created tenure_group")

    # ── 2. avg_monthly_spend ──────────────────────────────────────────────────
    df["avg_monthly_spend"] = df["TotalCharges"] / (tenure_numeric + 1)
    logger.info("Created avg_monthly_spend")

    # ── 3. num_services ───────────────────────────────────────────────────────
    # A service counts as subscribed if the value is 'Yes'
    def is_subscribed(val):
        if isinstance(val, str):
            return 1 if val.strip().lower() == "yes" else 0
        return 0

    df["num_services"] = sum(df[col].apply(is_subscribed) for col in ADDON_SERVICES)
    logger.info("Created num_services")

    # ── 4. has_internet ───────────────────────────────────────────────────────
    df["has_internet"] = (df["InternetService"].str.lower().fillna("no") != "no").astype(int)
    logger.info("Created has_internet")

    # ── 5. has_phone ──────────────────────────────────────────────────────────
    df["has_phone"] = df["PhoneService"].str.lower().str.contains("yes", na=False).astype(int)
    logger.info("Created has_phone")

    # ── 6. auto_pay ───────────────────────────────────────────────────────────
    df["auto_pay"] = df["PaymentMethod"].str.lower().str.contains("automatic", na=False).astype(int)
    logger.info("Created auto_pay")

    # ── 7. is_monthly_fiber ───────────────────────────────────────────────────
    is_monthly = df["Contract"].str.lower().str.contains("month-to-month", na=False)
    is_fiber   = df["InternetService"].str.lower().str.contains("fiber", na=False)
    df["is_monthly_fiber"] = (is_monthly & is_fiber).astype(int)
    logger.info("Created is_monthly_fiber")

    # ── 8. tenure_x_monthly ───────────────────────────────────────────────────
    df["tenure_x_monthly"] = tenure_numeric * df["MonthlyCharges"]
    logger.info("Created tenure_x_monthly")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN TRANSFORMER (used inside the Pipeline for training + serving)
# ─────────────────────────────────────────────────────────────────────────────

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that adds all engineered features.

    WHY we use a custom sklearn transformer:
      When we put this inside a Pipeline the same code runs during .fit()
      (training) and .predict() (serving).  This eliminates the risk of
      train/serve skew – a common production bug where the API creates
      features slightly differently from how they were created during training.

    Usage:
      pipe = Pipeline([
          ("feature_eng", ChurnFeatureEngineer()),
          ("preprocessor", ColumnTransformer(...)),
          ("classifier",   LogisticRegression()),
      ])
    """

    def fit(self, X, y=None):
        # No fitting needed – feature engineering is deterministic
        return self

    def transform(self, X):
        # X is a DataFrame; we call the same pure-pandas function
        df = pd.DataFrame(X)  # ensure it's a DataFrame (Pipeline may pass ndarray)
        return build_features_df(df)

    def get_feature_names_out(self, input_features=None):
        # Required for some sklearn meta-estimators
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE TARGET
# ─────────────────────────────────────────────────────────────────────────────

def encode_target(series: pd.Series) -> pd.Series:
    """
    Convert the 'Churn' column from 'Yes'/'No' to 1/0.

    WHY: sklearn classifiers require numeric labels.
    1 = churned (the positive / minority class we care most about).
    0 = stayed.
    """
    mapping = {"Yes": 1, "No": 0}
    encoded = series.map(mapping)
    if encoded.isnull().any():
        unknown = series[encoded.isnull()].unique()
        logger.warning(f"Unknown target values found: {unknown}")
    return encoded
