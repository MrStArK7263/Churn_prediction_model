"""
settings.py – reads configuration from environment variables.

WHY: Hard-coding paths and settings into the application is bad practice.
By reading from environment variables, the same Docker image can be used
in development (with local paths) and in production (with Render/AWS paths)
just by changing the environment variables.
"""

import os
from pathlib import Path

# ── Model path ────────────────────────────────────────────────────────────────
# On Render/AWS, set MODEL_PATH to point to the model file inside the container.
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "churn_model_v1.0.0.joblib"),
)

METADATA_PATH = os.getenv(
    "METADATA_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"),
)

# ── API settings ──────────────────────────────────────────────────────────────
MODEL_VERSION  = os.getenv("MODEL_VERSION", "1.0.0")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
PORT           = int(os.getenv("PORT", "8000"))

# ── CORS ──────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS to a comma-separated list of allowed frontend URLs.
# "*" allows any origin (fine for development, restrict in production).
_raw_origins   = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]
