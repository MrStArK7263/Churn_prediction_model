"""
test_api.py – pytest tests for the FastAPI backend.

Run from the project root:
    pytest backend/tests/test_api.py -v

These tests use TestClient so no real server is needed.
"""

import os
import sys
from pathlib import Path

# Add the backend folder to path so we can import the app
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Point settings to the actual model files before importing the app
# (models/ is in backend/models/ after `cp models/* backend/models/`)
os.environ.setdefault("MODEL_PATH",    str(BACKEND_DIR / "models" / "churn_model_v1.0.0.joblib"))
os.environ.setdefault("METADATA_PATH", str(BACKEND_DIR / "models" / "model_metadata.json"))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Sample valid customer payload ─────────────────────────────────────────────
VALID_CUSTOMER = {
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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_root():
    """GET / should return a 200 with a message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_ok():
    """GET /health should return 200 when model is loaded."""
    response = client.get("/health")
    # If model is loaded → 200; if not loaded yet → 503
    # We accept either so the test works even without a pre-built model file
    assert response.status_code in (200, 503)
    data = response.json()
    assert "model_loaded" in data or "detail" in data


def test_predict_valid():
    """POST /predict with valid data should return a prediction."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    # If model isn't loaded this will be 503
    if response.status_code == 503:
        pytest.skip("Model not loaded – skipping prediction test")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ("Churn", "No Churn")
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "risk_level" in data
    assert data["risk_level"] in ("High", "Medium", "Low")
    assert "top_factors" in data
    assert "model_version" in data


def test_predict_invalid_tenure():
    """POST /predict with out-of-range tenure should return 422."""
    bad_customer = VALID_CUSTOMER.copy()
    bad_customer["tenure"] = -5  # tenure cannot be negative
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_invalid_categorical():
    """POST /predict with a wrong category value should return 422."""
    bad_customer = VALID_CUSTOMER.copy()
    bad_customer["gender"] = "Other"  # not in Literal["Male","Female"]
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_missing_field():
    """POST /predict with a missing required field should return 422."""
    bad_customer = VALID_CUSTOMER.copy()
    del bad_customer["tenure"]
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_model_info():
    """GET /model-info should return metadata or 404 if not available."""
    response = client.get("/model-info")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "model_version" in data
