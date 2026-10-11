# Assignment Checklist – Customer Churn ML Project

This document maps every assignment requirement to its location in the project.

---

## ✅ 1 – Data Understanding
| Requirement | Location |
|-------------|----------|
| Rows and columns | `notebooks/churn_end_to_end.ipynb` → Section 1 |
| Numeric vs categorical columns | Section 1 – explicitly stated |
| SeniorCitizen is categorical | Section 1 + `src/data_cleaning.py::fix_data_types()` |
| Target column (`Churn`) | Section 1 – defined and explained |
| Churn / non-churn meaning | Section 1 markdown cell |
| Data types (`info()`) | Section 1 |
| Descriptive statistics | Section 1 – `describe()` and `value_counts` |
| Class distribution | Section 1 – counts and percentages |

---

## ✅ 2 – Data Cleaning
| Requirement | Location |
|-------------|----------|
| Missing value table + bar chart | Section 2 + `src/data_cleaning.py::missing_value_report()` |
| Drop 72 missing-target rows (with reason) | Section 2 + `drop_missing_target()` |
| Full-row duplicates detected + removed | Section 2 + `remove_duplicates()` |
| Duplicate customerIDs detected + removed | Section 2 + `remove_duplicates()` |
| `TotalCharges` text → numeric | `fix_data_types()` |
| `tenure` float → int | `fix_data_types()` |
| `SeniorCitizen` 0/1 → Yes/No | `fix_data_types()` |
| Outlier detection (IQR + box plots) | Section 2 |
| Negative/invalid numeric values | `fix_invalid_numeric_values()` |
| Inconsistent categorical values | `clean_categorical_values()` – strip + title case |
| Drop `customerID` with reason | `drop_unnecessary_columns()` |
| Leakage detection discussion | Section 2 – TotalCharges ≈ tenure × MonthlyCharges analysis |
| Imputation inside Pipeline | `src/train.py::build_preprocessor()` |
| Reason for each cleaning step | Docstrings in `src/data_cleaning.py` + notebook markdown cells |

---

## ✅ 3 – EDA
| Requirement | Location |
|-------------|----------|
| Target class distribution | Section 3 |
| Histograms and KDE | Section 3 |
| Box plots (overall + by churn) | Section 3 |
| Count plots for categoricals | Section 3 |
| Churn rate % per category | Section 3 |
| Feature vs target (box/violin + stacked bars) | Section 3 |
| Outlier analysis visuals | Section 3 |
| Key business insights | Section 3 – markdown summary cell |
| Observations under each chart | Section 3 – markdown "Observations:" cells |

---

## ✅ 4 – Correlation Analysis
| Requirement | Location |
|-------------|----------|
| Correlation matrix + heatmap | Section 4 |
| Highly correlated pairs (\|r\| > 0.8) | Section 4 |
| Negatively correlated features | Section 4 |
| Little/no correlation features | Section 4 |
| VIF computation + explanation | Section 4 |
| Decision to keep/remove TotalCharges | Section 4 – justified in markdown |

---

## ✅ 5 – Feature Engineering
| Requirement | Location |
|-------------|----------|
| `tenure_group` binning | `src/feature_engineering.py::build_features_df()` |
| `avg_monthly_spend` ratio | `build_features_df()` |
| `num_services` count | `build_features_df()` |
| `has_internet`, `has_phone` flags | `build_features_df()` |
| `auto_pay` flag | `build_features_df()` |
| `is_monthly_fiber` interaction | `build_features_df()` |
| `tenure_x_monthly` interaction | `build_features_df()` |
| Target encoding (Yes→1, No→0) | `encode_target()` |
| One-hot encoding (nominal) | `build_preprocessor()` → `OneHotEncoder` |
| Ordinal encoding (Contract) | `build_preprocessor()` → `OrdinalEncoder` |
| `StandardScaler` for numeric | `build_preprocessor()` → `StandardScaler` |
| Reusable sklearn transformer | `ChurnFeatureEngineer` (used by both train + backend) |
| Date extraction | ⚠️ **Not applicable** – the dataset has no date columns (see note) |

> **Note on date features:** The assignment mentions date extraction as an example feature engineering step. The `data_23.csv` dataset does not contain any date/timestamp columns (columns are: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn). Therefore, date extraction cannot be performed and this sub-task is explicitly noted as not applicable.

---

## ✅ 6 – Feature Selection
| Requirement | Location |
|-------------|----------|
| Correlation analysis | Section 6 |
| Random Forest feature importance | Section 6 |
| Mutual information scores | Section 6 |
| RFE with Logistic Regression | Section 6 |
| `SelectFromModel` | Section 6 |
| Ranked feature table / plot | Section 6 |
| Before vs after feature selection comparison | Section 6 – table with all metrics |

---

## ✅ 7 – Train/Test Split
| Requirement | Location |
|-------------|----------|
| 80/20 stratified split, random_state=42 | `src/train.py` + Section 7 |
| Class balance in both sets printed | Section 7 |
| 5-fold stratified cross-validation | `src/train.py::train()` + Section 7 |

---

## ✅ 8 – Multiple Models
| Requirement | Location |
|-------------|----------|
| Logistic Regression | `src/train.py::build_models()` + Section 8 |
| Decision Tree | `src/train.py::build_models()` + Section 8 |
| Random Forest | `src/train.py::build_models()` + Section 8 |
| class_weight="balanced" with explanation | `build_models()` docstring + Section 8 |
| Deployed model is from this list | Final model saved to `models/churn_model_v1.0.0.joblib` |

---

## ✅ 9 – Evaluation
| Requirement | Location |
|-------------|----------|
| Confusion matrix (plotted for each model) | Section 9 |
| Accuracy | `compute_metrics()` + Section 9 |
| Precision | `compute_metrics()` + Section 9 |
| Recall | `compute_metrics()` + Section 9 |
| F1-score | `compute_metrics()` + Section 9 |
| ROC-AUC + overlaid ROC curves | Section 9 |
| Train vs test scores (overfitting check) | Section 9 |
| Summary comparison table | Section 9 |
| Why accuracy is not enough | Section 9 – markdown explanation |

---

## ✅ 10 – Hyperparameter Tuning
| Requirement | Location |
|-------------|----------|
| GridSearchCV on best 2 models | `src/train.py` + Section 10 |
| Before/after comparison table with % improvement | Section 10 |
| Scoring = F1 or recall on churn class | `GridSearchCV(scoring="f1")` in train.py |

---

## ✅ 11 – Final Model Justification
| Requirement | Location |
|-------------|----------|
| Comparison: accuracy, F1, recall, ROC-AUC, inference time, size, complexity | Section 11 |
| Written justification paragraph | Section 11 – "Why did I select this model?" |

---

## ✅ 12 – Save Model
| Requirement | Location |
|-------------|----------|
| Full Pipeline saved with joblib | `models/churn_model_v1.0.0.joblib` |
| `model_metadata.json` with all fields | `models/model_metadata.json` |
| Reload + sample prediction in notebook | Section 12 |
| `src/train.py` reproduces training from CLI | `python src/train.py` |

---

## ✅ 13 – API (FastAPI)
| Requirement | Location |
|-------------|----------|
| `GET /` | `backend/app/main.py` |
| `GET /health` | `backend/app/main.py` → returns 503 if model failed |
| `GET /model-info` | `backend/app/main.py` |
| `POST /predict` with full response schema | `backend/app/main.py` + `schemas.py` |
| `POST /predict-batch` (optional) | `backend/app/main.py` |
| Pydantic validation (Literal, ranges, 422 errors) | `backend/app/schemas.py` |
| Error handling (custom handler, HTTP codes) | `generic_exception_handler` in `main.py` |
| Environment variables | `backend/app/settings.py` |
| Model versioning in every response | `predictor.py` + `schemas.py` |
| Load model once at startup | `lifespan` in `main.py` + `predictor.py::load_model()` |
| CORS enabled | `CORSMiddleware` in `main.py` |
| Swagger docs at `/docs` | Built into FastAPI |
| pytest tests | `backend/tests/test_api.py` |

---

## ✅ 14 – Web UI (Streamlit)
| Requirement | Location |
|-------------|----------|
| Input fields for every feature | `frontend/app.py` – form with selectboxes + sliders |
| Grouped sections (Demographics, Services, Billing) | `frontend/app.py` |
| Predict button | `frontend/app.py` – `st.form_submit_button` |
| Prediction class (colour-coded) | `frontend/app.py` – red/green boxes |
| Probability bar | `frontend/app.py` – `st.progress()` |
| Risk level | `frontend/app.py` – metric display |
| Top factors | `frontend/app.py` – factor list |
| API status in sidebar | `frontend/app.py` – sidebar calls `/health` |
| Model version in sidebar | `frontend/app.py` – sidebar calls `/model-info` |
| `API_URL` from env var | `frontend/app.py` – `os.getenv("API_URL")` |
| Error messages (API down, 422, timeout) | `frontend/app.py` – try/except with user-friendly messages |
| Spinner while waiting | `frontend/app.py` – `st.spinner()` |

---

## ✅ 15 – Deployment
| Requirement | Location |
|-------------|----------|
| Docker (backend) | `backend/Dockerfile` |
| Docker (frontend, optional) | `frontend/Dockerfile` |
| docker-compose (local) | `docker-compose.yml` |
| Render deployment | `render.yaml` + `docs/DEPLOY_RENDER.md` |
| AWS guide (ECR + App Runner / EC2) | `docs/DEPLOY_AWS.md` |
| Architecture diagram | `docs/ARCHITECTURE.md` |

---

## ✅ 16 – Production Quality
| Requirement | Location |
|-------------|----------|
| Input validation | `backend/app/schemas.py` – Pydantic |
| Error handling | `backend/app/main.py` – `generic_exception_handler` |
| `requirements.txt` (backend, pinned) | `backend/requirements.txt` |
| `requirements.txt` (frontend, minimal) | `frontend/requirements.txt` |
| `requirements.txt` (training) | `requirements.txt` (root) |
| Environment variables | `backend/app/settings.py` + `.env.example` |
| Model versioning | `src/config.py::MODEL_VERSION` + metadata + responses |
| `/health` endpoint | `backend/app/main.py` |
| Dockerfile (non-root, healthcheck, layer caching) | `backend/Dockerfile` |
| `.gitignore` (venv, .env, __pycache__) | `.gitignore` |
| `README.md` | `README.md` |
