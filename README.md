# 📡 Customer Churn Prediction – End-to-End ML Project

> Predict whether a telecom customer will churn using machine learning.
> Built as a complete production-style pipeline: data cleaning → EDA → feature engineering →
> model training → REST API (FastAPI) → Web UI (Streamlit) → Docker → Render deployment.

---

## 🗂️ Project Structure

```
churn-ml-project/
├── data/                         # Raw dataset
│   └── data_23.csv
├── notebooks/
│   └── churn_end_to_end.ipynb    # Full analysis (12 sections)
├── src/
│   ├── config.py                 # Paths and constants
│   ├── data_cleaning.py          # Reusable cleaning functions
│   ├── feature_engineering.py    # Reusable feature functions + sklearn transformer
│   └── train.py                  # CLI training script
├── models/
│   ├── churn_model_v1.0.0.joblib # Saved pipeline
│   └── model_metadata.json       # Version, metrics, features
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI endpoints
│   │   ├── schemas.py            # Pydantic input/output validation
│   │   ├── predictor.py          # Model loading + prediction
│   │   └── settings.py           # Environment variable config
│   ├── tests/test_api.py         # pytest tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                    # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOY_RENDER.md
│   ├── DEPLOY_AWS.md
│   └── ASSIGNMENT_CHECKLIST.md
├── docker-compose.yml
├── render.yaml
└── requirements.txt              # Training dependencies
```

---

## 📊 Dataset

- **File:** `data/data_23.csv` — ~7,111 rows × 21 columns
- **Source:** Telecom customer records
- **Target:** `Churn` (Yes / No)
- **Class balance:** ~73% No Churn, ~27% Churn (imbalanced)

Key messiness handled:
- 72 rows with missing `Churn` target → dropped
- `TotalCharges` stored as text → converted with `pd.to_numeric`
- ~20 duplicate rows + duplicate customerIDs → removed
- `SeniorCitizen` stored as 0/1 → converted to Yes/No category
- Inconsistent whitespace/case in categorical columns → standardised

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** ★ | 0.7507 | 0.5208 | **0.7721** | 0.6220 | **0.8433** |
| Random Forest (tuned) | 0.7934 | 0.5488 | 0.7373 | 0.6279 | 0.8256 |
| Decision Tree | 0.7286 | 0.4617 | 0.5228 | 0.5058 | 0.6629 |

> ⚡ Re-run `python src/train.py` at any time to regenerate these results.

**Final model:** Logistic Regression (tuned, C=1, solver=lbfgs)

**Why not pick purely by accuracy?**
A model that always predicts "No Churn" gets ~73% accuracy but catches
zero churners. We optimise for **Recall** (catching actual churners)
and **F1** (balance of precision and recall).

---

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → Preprocessing Pipeline → ML Model → Prediction + Probability
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram.

---

## 🚀 How to Run Locally

### Option 1: Python (venv)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install training dependencies
pip install -r requirements.txt

# 3. Train the model (creates models/ files)
python src/train.py

# 4. Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# 5. Start the FastAPI backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. In a new terminal: start the Streamlit frontend
pip install streamlit requests
streamlit run frontend/app.py
```

Open:
- Backend API docs: http://localhost:8000/docs
- Frontend UI:      http://localhost:8501

### Option 2: Docker Compose

```bash
# Make sure the model is already trained first:
python src/train.py

# Copy the model to the backend's models folder
cp models/churn_model_v1.0.0.joblib backend/models/
cp models/model_metadata.json backend/models/

# Build and start both services
docker compose up --build

# Open http://localhost:8501 for the UI
# Open http://localhost:8000/docs for the API
```

---

## 🔌 API Examples

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Predict churn for one customer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 29.85
  }'
```

---

## 🧪 Run Tests

```bash
cd backend
C:\Users\talas\anaconda3\python.exe -m pytest tests/test_api.py -v
```

---

## 🌐 Deploy to Render

See [`docs/DEPLOY_RENDER.md`](docs/DEPLOY_RENDER.md) for step-by-step instructions.

Short version:
1. Push to GitHub
2. Connect repo to Render → New Web Service → Runtime: Docker
3. Set root dir `backend`, health check `/health`
4. Deploy frontend as a second service with `API_URL` set to the backend URL

---

## ☁️ Deploy to AWS

See [`docs/DEPLOY_AWS.md`](docs/DEPLOY_AWS.md) for ECR + App Runner guide.

---

## ⚙️ Retrain the Model

```bash
# Just run this – it handles everything
python src/train.py

# Then update the backend model files
cp models/churn_model_v1.0.0.joblib backend/models/
cp models/model_metadata.json backend/models/
```

> ⚠️ **Important:** The `scikit-learn` version in `backend/requirements.txt`
> must match the version used for training. Joblib model files are **not**
> compatible across major scikit-learn versions. If you update sklearn,
> update `backend/requirements.txt` too and retrain.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn 1.2.2, pandas, numpy |
| Notebook | Jupyter |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| UI | Streamlit |
| Serialisation | joblib |
| Container | Docker |
| Deployment | Render (primary), AWS ECR + App Runner (secondary) |
| Version control | GitHub |

---

## 📸 Screenshots

> _Add screenshots after deployment_

- [ ] Streamlit UI – input form
- [ ] Streamlit UI – prediction result
- [ ] FastAPI Swagger docs (/docs)
- [ ] Render dashboard

---

## 🔗 Deployment Links

> _Fill in after deployment_

- **Backend API:** `https://YOUR_APP.onrender.com`
- **Frontend UI:** `https://YOUR_APP.onrender.com` or Streamlit Community Cloud

---

## 📋 Assignment Coverage

See [`docs/ASSIGNMENT_CHECKLIST.md`](docs/ASSIGNMENT_CHECKLIST.md) for a complete
mapping of every assignment requirement to its location in the project.
