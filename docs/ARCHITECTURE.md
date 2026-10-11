# System Architecture

## Overview

This project follows a standard **train → serve → display** ML architecture.

```
                       ┌─────────────────────────────────────┐
                       │            TRAINING PHASE            │
                       │                                      │
  data/data_23.csv ───►│  src/data_cleaning.py               │
                       │  src/feature_engineering.py         │
                       │  src/train.py                        │
                       │            │                         │
                       │            ▼                         │
                       │  models/churn_model_v1.0.0.joblib   │
                       │  models/model_metadata.json         │
                       └─────────────────────────────────────┘

                       ┌─────────────────────────────────────┐
                       │            SERVING PHASE             │
                       │                                      │
  User → Browser  ────►│  Streamlit UI (port 8501)           │
                       │  frontend/app.py                     │
                       │            │                         │
                       │   HTTP POST /predict                 │
                       │            │                         │
                       │            ▼                         │
                       │  FastAPI Backend (port 8000)         │
                       │  backend/app/main.py                 │
                       │            │                         │
                       │            ▼                         │
                       │  Preprocessing Pipeline              │
                       │  (ColumnTransformer: impute/scale/  │
                       │   encode)                            │
                       │            │                         │
                       │            ▼                         │
                       │  ML Model (sklearn Pipeline)         │
                       │  churn_model_v1.0.0.joblib          │
                       │            │                         │
                       │            ▼                         │
                       │  Prediction + Probability + Risk    │
                       └─────────────────────────────────────┘
```

## Request Flow (detailed)

```
User fills form
    │
    ▼
Streamlit app.py
  - Validates inputs locally (number ranges, dropdown choices)
  - Calls POST http://API_URL/predict with JSON body
    │
    ▼
FastAPI main.py
  - Pydantic validates the request (type check + range check)
  - Returns 422 if invalid
    │
    ▼
predictor.py
  - Converts dict → pandas DataFrame
  - Calls pipeline.predict_proba(df)
    │
    ▼
sklearn Pipeline
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: ChurnFeatureEngineer (custom transformer)   │
  │   - Adds tenure_group, num_services, auto_pay, etc. │
  │                                                     │
  │ Step 2: ColumnTransformer                           │
  │   - Numeric: SimpleImputer(median) + StandardScaler │
  │   - Nominal cat: SimpleImputer(mode) + OneHotEncoder│
  │   - Contract: OrdinalEncoder                        │
  │                                                     │
  │ Step 3: Classifier (e.g., Random Forest)            │
  │   - Returns [prob_no_churn, prob_churn]             │
  └─────────────────────────────────────────────────────┘
    │
    ▼
predictor.py
  - Applies threshold (default 0.5)
  - Determines risk level (High/Medium/Low)
  - Builds top_factors explanation
  - Returns JSON response
    │
    ▼
Streamlit displays:
  - Prediction verdict (Churn / No Churn)
  - Probability bars
  - Risk badge
  - Top factors list
```

## Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `src/data_cleaning.py` | Reusable cleaning functions (missing values, dtypes, duplicates) |
| `src/feature_engineering.py` | Engineered features + sklearn transformer |
| `src/train.py` | End-to-end training pipeline (runs from CLI) |
| `src/config.py` | Central paths and constants |
| `backend/app/main.py` | FastAPI routes and middleware |
| `backend/app/schemas.py` | Pydantic input/output validation |
| `backend/app/predictor.py` | Model loading and prediction logic |
| `backend/app/settings.py` | Environment variable handling |
| `frontend/app.py` | Streamlit UI |
| `notebooks/churn_end_to_end.ipynb` | Full analysis and model exploration |

## Technology Choices

| Decision | Choice | Why |
|----------|--------|-----|
| ML framework | scikit-learn | Well-documented, beginner-friendly, production-stable |
| Feature pipeline | sklearn Pipeline | Prevents train/serve skew |
| API framework | FastAPI | Automatic Swagger docs, Pydantic validation, async support |
| UI | Streamlit | Fast to build, no frontend JS needed |
| Serialisation | joblib | Standard for sklearn models, handles numpy arrays |
| Container | Docker | Reproducible, deployable anywhere |
| Deployment | Render | Free tier, Docker native, GitHub CI/CD |
