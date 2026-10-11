"""
main.py – FastAPI application entry point.

Endpoints:
  GET  /           → welcome message
  GET  /health     → health check (used by Render)
  GET  /model-info → model metadata
  POST /predict    → churn prediction for one customer
  POST /predict-batch → churn prediction for multiple customers (optional)
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import CustomerFeatures, PredictionResponse, HealthResponse
from .predictor import load_model, predict, is_model_loaded, get_metadata
from .settings import ALLOWED_ORIGINS, MODEL_VERSION, LOG_LEVEL

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load model ONCE at startup ──────────────────────────────────────
# WHY lifespan instead of @app.on_event:
#   The lifespan context manager is the modern FastAPI way (v0.93+).
#   It loads the model before the first request arrives.

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up – loading model...")
    load_model()
    if is_model_loaded():
        logger.info("Model loaded and ready")
    else:
        logger.error("Model failed to load – /health will return 503")
    yield
    logger.info("Shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict whether a telecom customer will churn. "
        "Built with FastAPI + scikit-learn. "
        "See /docs for interactive documentation."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# WHY CORS: The Streamlit frontend runs on a different port/host.
# Without CORS the browser blocks requests from the frontend to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handler ───────────────────────────────────────────────────────
# WHY: Catching unhandled exceptions here means we can return a clean JSON
# error message instead of leaking a Python stack trace to the client.
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Welcome message."""
    return {
        "message": "Customer Churn Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """
    Health check endpoint used by Render and load balancers.
    Returns 503 if the model failed to load so Render knows not to route
    traffic to this instance.
    """
    loaded = is_model_loaded()
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "model_loaded": False, "model_version": MODEL_VERSION},
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION,
    )


@app.get("/model-info", tags=["General"])
def model_info():
    """Return the full contents of model_metadata.json."""
    meta = get_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata not found")
    return meta


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer.

    Returns:
      - prediction: "Churn" or "No Churn"
      - churn_probability: float 0–1
      - risk_level: "High" / "Medium" / "Low"
      - top_factors: list of explanation dicts
      - model_version
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"/predict error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check your input and try again.",
        )


@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(customers: List[CustomerFeatures]):
    """
    Predict churn for a list of customers (optional endpoint).
    Returns a list of PredictionResponse objects.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for i, customer in enumerate(customers):
        try:
            result = predict(customer.model_dump())
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error for index {i}: {e}")
            results.append({"error": str(e), "index": i})

    return {"predictions": results, "count": len(results)}
