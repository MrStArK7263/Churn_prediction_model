"""conftest.py – pytest configuration, runs before any test file."""
import os
import sys
from pathlib import Path

# Make backend importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Point env vars at the real model files
os.environ.setdefault("MODEL_PATH",    str(BACKEND_DIR / "models" / "churn_model_v1.0.0.joblib"))
os.environ.setdefault("METADATA_PATH", str(BACKEND_DIR / "models" / "model_metadata.json"))

# Pre-load the model so TestClient tests don't need the lifespan to run
from app.predictor import load_model
load_model()
