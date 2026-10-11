"""
predictor.py – loads the model once at startup and makes predictions.

WHY load once at startup:
  Loading a joblib file from disk takes ~50-200ms.
  If we loaded it for every request the API would be very slow.
  By loading it once when the server starts, every request uses the
  already-loaded in-memory object (~1-5ms).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ── Make src/ importable so we can reuse the same feature engineering ─────────
# The backend/ folder is one level above this file; project root is two levels up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.feature_engineering import build_features_df
from src.data_cleaning import fix_data_types, clean_categorical_values

from .settings import MODEL_PATH, METADATA_PATH, MODEL_VERSION

logger = logging.getLogger(__name__)

# ── Global variables (loaded once at startup) ─────────────────────────────────
_model = None          # the sklearn Pipeline
_metadata = {}         # contents of model_metadata.json
_model_loaded = False  # flag used by /health


def load_model():
    """
    Load the model and metadata from disk.
    Called once when FastAPI starts (in the lifespan context).
    If this fails, _model_loaded stays False and /health returns 503.
    """
    global _model, _metadata, _model_loaded

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model_loaded = False
        return

    try:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        logger.info("Metadata loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
        _metadata = {"model_version": MODEL_VERSION}

    _model_loaded = True


def is_model_loaded() -> bool:
    return _model_loaded


def get_metadata() -> dict:
    return _metadata


# ── Feature importance helper ─────────────────────────────────────────────────

# Map of feature names to plain-English labels (for the response)
FEATURE_LABELS = {
    "Contract":           "Contract type",
    "InternetService":    "Internet service type",
    "tenure":             "How long with the company",
    "MonthlyCharges":     "Monthly bill amount",
    "TotalCharges":       "Total charges paid",
    "PaymentMethod":      "Payment method",
    "PaperlessBilling":   "Paperless billing",
    "OnlineSecurity":     "Online security add-on",
    "TechSupport":        "Tech support add-on",
    "is_monthly_fiber":   "Month-to-month + Fiber Optic (highest risk combo)",
    "num_services":       "Number of add-on services",
    "avg_monthly_spend":  "Average monthly spend",
    "auto_pay":           "Auto-pay enabled",
    "tenure_group":       "Tenure group",
    "tenure_x_monthly":   "Tenure × Monthly Charges",
}

# Business rules for quick explanations (shown in the response)
def _explain_factors(customer: dict, prob: float) -> list:
    """
    Return a list of {feature, impact} dicts that explain the prediction.

    WHY: Instead of complex SHAP values (which require extra libraries),
    we use simple business rules derived from EDA insights.
    This keeps the backend lightweight while still giving useful context.
    """
    factors = []

    # Contract type is the strongest predictor
    if customer.get("Contract") == "Month-To-Month":
        factors.append({
            "feature": "Contract",
            "impact": "Month-to-month contracts are the highest churn risk. Consider offering a discount for 1-year contract."
        })
    elif customer.get("Contract") == "Two Year":
        factors.append({
            "feature": "Contract",
            "impact": "Two-year contract significantly reduces churn risk."
        })

    # Short tenure = high risk
    tenure = customer.get("tenure", 0)
    if tenure <= 12:
        factors.append({
            "feature": "tenure",
            "impact": f"Only {tenure} months with the company. New customers are much more likely to churn."
        })
    elif tenure >= 48:
        factors.append({
            "feature": "tenure",
            "impact": f"{tenure} months tenure – long-term customers rarely churn."
        })

    # Fiber optic internet + month-to-month
    if (customer.get("InternetService", "").lower() == "fiber optic"
            and customer.get("Contract") == "Month-To-Month"):
        factors.append({
            "feature": "InternetService + Contract",
            "impact": "Fiber Optic + Month-to-Month is the highest-churn segment in the data."
        })

    # Electronic check payment
    if customer.get("PaymentMethod") == "Electronic Check":
        factors.append({
            "feature": "PaymentMethod",
            "impact": "Electronic check payers have higher churn than auto-pay customers."
        })

    # No tech support
    if customer.get("TechSupport") == "No":
        factors.append({
            "feature": "TechSupport",
            "impact": "Customers without tech support tend to churn more often."
        })

    # High monthly charges
    monthly = customer.get("MonthlyCharges", 0)
    if monthly > 80:
        factors.append({
            "feature": "MonthlyCharges",
            "impact": f"High monthly bill (${monthly:.2f}) increases churn risk."
        })

    # If no factors found, add a generic one
    if not factors:
        factors.append({
            "feature": "Overall profile",
            "impact": f"Churn probability: {prob:.1%}. Customer profile is relatively low risk.",
        })

    # Return top 3
    return factors[:3]


# ── Main prediction function ──────────────────────────────────────────────────

def predict(customer_data: dict) -> dict:
    """
    Run the full prediction pipeline for one customer.

    Steps:
      1. Convert the dict to a DataFrame
      2. Apply the same type fixes + categorical cleaning done in training
      3. Apply feature engineering (same function used during training)
      4. Run model.predict_proba() to get probabilities
      5. Apply decision threshold, determine risk level, build explanation
      6. Return structured response
    """
    if not _model_loaded:
        raise RuntimeError("Model is not loaded")

    # Step 1 – Build a single-row DataFrame
    df = pd.DataFrame([customer_data])

    # Step 2 – Ensure tenure is numeric (API sends it as int already, but be safe)
    df["tenure"]        = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"]= pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # SeniorCitizen comes as "Yes"/"No" from the API schema (Pydantic enforced)
    # so we do NOT call fix_data_types (which expects 0/1 numeric input)

    # Step 3 – Standardise categorical strings (strip + title case)
    df = clean_categorical_values(df)

    # Step 3 – Feature engineering (same function as training)
    df = build_features_df(df)

    # Drop target if accidentally present (shouldn't happen at serve time)
    df = df.drop(columns=["Churn"], errors="ignore")

    # Step 4 – Predict via the fitted sklearn Pipeline
    proba = _model.predict_proba(df)[0]
    churn_prob     = float(proba[1])
    non_churn_prob = float(proba[0])

    # Step 5 – Apply threshold from metadata (default 0.5)
    threshold        = _metadata.get("threshold", 0.5)
    prediction_label = int(churn_prob >= threshold)
    prediction_str   = "Churn" if prediction_label == 1 else "No Churn"

    # Risk level
    if churn_prob >= 0.70:
        risk_level = "High"
    elif churn_prob >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Explanation factors
    factors = _explain_factors(customer_data, churn_prob)

    return {
        "prediction":            prediction_str,
        "prediction_label":      prediction_label,
        "churn_probability":     round(churn_prob, 4),
        "non_churn_probability": round(non_churn_prob, 4),
        "risk_level":            risk_level,
        "top_factors":           factors,
        "model_version":         _metadata.get("model_version", MODEL_VERSION),
    }
