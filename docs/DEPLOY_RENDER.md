# Deploying to Render – Step-by-Step Guide (Beginner Friendly)

## What is Render?
Render is a cloud platform where you can run web applications for free
(with a few limitations on the free tier). We will deploy two services:
1. **Backend** – the FastAPI API (Docker container)
2. **Frontend** – the Streamlit UI (Docker container)

---

## Step 1 – Push the project to GitHub

1. Create a new **GitHub** repository at https://github.com/new
2. From your local project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
3. Make sure the `models/` folder is committed (the model file must be
   in the repository so Render can build the Docker image without retraining).

---

## Step 2 – Deploy the Backend API

1. Go to **https://render.com** and create a free account (or log in).
2. Click **New → Web Service**.
3. Connect your **GitHub** account and select your repository.
4. Fill in the settings:
   - **Name:** `churn-backend`
   - **Runtime:** Docker
   - **Dockerfile Path:** `./backend/Dockerfile`
   - **Docker Context:** `./backend`
   - **Health Check Path:** `/health`
5. Click **Advanced** and add these **Environment Variables**:
   | Key | Value |
   |-----|-------|
   | `MODEL_VERSION` | `1.0.0` |
   | `LOG_LEVEL` | `INFO` |
   | `ALLOWED_ORIGINS` | `*` (change after you know the frontend URL) |
6. Click **Deploy**. Render will build the Docker image and start the service.
7. After deployment, visit `https://churn-backend.onrender.com/health` to
   confirm the API is running.
8. Copy the backend URL (e.g., `https://churn-backend.onrender.com`) – you'll
   need it for the frontend.

---

## Step 3 – Deploy the Frontend

**Option A: Render Docker service (same repo)**
1. Click **New → Web Service** again.
2. Same repository, different settings:
   - **Name:** `churn-frontend`
   - **Runtime:** Docker
   - **Dockerfile Path:** `./frontend/Dockerfile`
   - **Docker Context:** `./frontend`
3. Add **Environment Variable:**
   | Key | Value |
   |-----|-------|
   | `API_URL` | `https://churn-backend.onrender.com` |
4. Click **Deploy**.
5. Visit `https://churn-frontend.onrender.com` to use the UI.

**Option B: Streamlit Community Cloud (easiest, free)**
1. Go to **https://share.streamlit.io** and log in with GitHub.
2. Click **New app** → select your repository and branch.
3. Set **Main file path:** `frontend/app.py`
4. Add a **Secret** (or environment variable): `API_URL = https://churn-backend.onrender.com`
5. Deploy.

---

## Step 4 – Test

1. Open `https://YOUR_BACKEND.onrender.com/docs` – you should see the
   Swagger UI with all endpoints.
2. Try the `/health` endpoint – it should return `{"status": "ok", ...}`.
3. Open the Streamlit frontend URL and make a prediction.

---

## ⚠️ Free Tier Notes

- **Cold starts:** Free Render services spin down after 15 minutes of
  inactivity. The first request after a cold start may take **20-60 seconds**.
  This is normal on the free tier.
- **Memory:** Free tier has 512 MB RAM. The Random Forest model uses ~50-100 MB.
- **Disk:** The model file (~5-10 MB) must be committed to the repo.

---

## Updating the Model

If you retrain and get a new model file:
1. Update `MODEL_VERSION` in `src/config.py` and in `backend/requirements.txt`.
2. Copy the new `.joblib` file to `backend/models/`.
3. Push to GitHub → Render redeploys automatically.
