"""
schemas.py – Pydantic models for request/response validation.

WHY Pydantic:
  FastAPI uses Pydantic to automatically validate incoming JSON.
  If the user sends a wrong type (e.g., tenure = "abc") or an out-of-range
  value, FastAPI returns a clear 422 error with a list of what went wrong,
  before the code even touches the model.  This is much safer than manual
  if/else checks.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ── Allowed categorical values (mirrors the training data) ────────────────────
GenderType          = Literal["Male", "Female"]
YesNoType           = Literal["Yes", "No"]
MultipleLinesType   = Literal["Yes", "No", "No Phone Service"]
InternetServiceType = Literal["DSL", "Fiber Optic", "No"]
AddonServiceType    = Literal["Yes", "No", "No Internet Service"]
ContractType        = Literal["Month-To-Month", "One Year", "Two Year"]
PaymentMethodType   = Literal[
    "Electronic Check",
    "Mailed Check",
    "Bank Transfer (Automatic)",
    "Credit Card (Automatic)",
]


# ── Input model ───────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    All features needed to predict churn for one customer.
    Pydantic enforces types and ranges automatically.
    """
    gender:           GenderType
    SeniorCitizen:    YesNoType
    Partner:          YesNoType
    Dependents:       YesNoType
    tenure:           int   = Field(..., ge=0, le=72, description="Number of months with the company (0–72)")
    PhoneService:     YesNoType
    MultipleLines:    MultipleLinesType
    InternetService:  InternetServiceType
    OnlineSecurity:   AddonServiceType
    OnlineBackup:     AddonServiceType
    DeviceProtection: AddonServiceType
    TechSupport:      AddonServiceType
    StreamingTV:      AddonServiceType
    StreamingMovies:  AddonServiceType
    Contract:         ContractType
    PaperlessBilling: YesNoType
    PaymentMethod:    PaymentMethodType
    MonthlyCharges:   float = Field(..., gt=0,  description="Monthly bill amount (must be > 0)")
    TotalCharges:     float = Field(..., ge=0,  description="Total amount charged (must be ≥ 0)")

    class Config:
        # Allow extra fields to be ignored (future-proofing)
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No Phone Service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-To-Month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic Check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }


# ── Feature factor (for explanation) ─────────────────────────────────────────

class FeatureFactor(BaseModel):
    feature: str
    impact: str


# ── Output model ──────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction:           str         # "Churn" or "No Churn"
    prediction_label:     int         # 1 or 0
    churn_probability:    float       # probability of churn
    non_churn_probability: float      # 1 - churn_probability
    risk_level:           str         # "High", "Medium", "Low"
    top_factors:          List[FeatureFactor]
    model_version:        str


# ── Health response ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
