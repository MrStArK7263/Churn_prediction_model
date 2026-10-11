"""
app.py – Streamlit frontend for the Churn Prediction API.

This is a separate app from the backend.  It calls the FastAPI /predict
endpoint over HTTP using the `requests` library.

The API_URL is read from an environment variable or st.secrets so it works
locally AND on Render without changing any code.

Run locally:
    streamlit run frontend/app.py
"""

import os
import time

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor – TelecoML",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API URL ───────────────────────────────────────────────────────────────────
# Priority: environment variable → st.secrets → default (localhost)
API_URL = (
    os.getenv("API_URL")
    or st.secrets.get("API_URL", None) if hasattr(st, "secrets") else None
    or "http://localhost:8000"
)
REQUEST_TIMEOUT = 15  # seconds

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #4F8BF9; }
    .subtitle   { color: #888; font-size: 1.1rem; margin-bottom: 1rem; }
    .churn-box  { background: #FF4B4B22; border-left: 4px solid #FF4B4B;
                  padding: 1rem; border-radius: 8px; }
    .nochurn-box{ background: #21C35422; border-left: 4px solid #21C354;
                  padding: 1rem; border-radius: 8px; }
    .factor-item{ background: #F0F2F6; padding: 0.5rem 0.75rem;
                  border-radius: 6px; margin-bottom: 0.4rem; font-size: 0.9rem; }
    .section-header { font-size: 1.1rem; font-weight: 700;
                      color: #4F8BF9; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")

    # API status (calls /health)
    st.markdown("### API Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success(f"✅ Online — Model v{data.get('model_version', '?')}")
        else:
            st.error(f"⚠️ API returned {health_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Is the backend running?")
    except Exception as e:
        st.warning(f"⚠️ {e}")

    # Model info
    st.markdown("---")
    st.markdown("### Model Info")
    try:
        info_resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if info_resp.status_code == 200:
            meta = info_resp.json()
            metrics = meta.get("metrics", {})
            st.markdown(f"**Model:** {meta.get('model_name', '?')}")
            st.markdown(f"**Version:** {meta.get('model_version', '?')}")
            st.markdown(f"**Trained:** {meta.get('trained_at', '?')[:10]}")
            if metrics:
                st.markdown("**Test Metrics:**")
                st.markdown(f"- Accuracy: `{metrics.get('accuracy', '?')}`")
                st.markdown(f"- F1 Score: `{metrics.get('f1', '?')}`")
                st.markdown(f"- Recall: `{metrics.get('recall', '?')}`")
                st.markdown(f"- ROC-AUC: `{metrics.get('roc_auc', '?')}`")
    except Exception:
        st.info("Model info unavailable")

    st.markdown("---")
    st.caption(f"API: `{API_URL}`")


# ── Main page ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📡 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter a customer\'s details below to predict whether they are likely to churn. '
    'All fields are required.</p>',
    unsafe_allow_html=True,
)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    with col2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    with col3:
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
    with col4:
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")

    # ── Services ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📶 Services</p>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No Phone Service"],
            key="multilines",
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber Optic", "No"],
            key="internet",
        )
    with col6:
        online_sec  = st.selectbox("Online Security",   ["No", "Yes", "No Internet Service"], key="osec")
        online_back = st.selectbox("Online Backup",     ["No", "Yes", "No Internet Service"], key="oback")
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No Internet Service"], key="dprot")

    col7, col8, col9 = st.columns(3)
    with col7:
        tech_support = st.selectbox("Tech Support",     ["No", "Yes", "No Internet Service"], key="tech")
    with col8:
        streaming_tv = st.selectbox("Streaming TV",     ["No", "Yes", "No Internet Service"], key="stv")
    with col9:
        streaming_mv = st.selectbox("Streaming Movies", ["No", "Yes", "No Internet Service"], key="smv")

    # ── Billing / Contract ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">💳 Billing & Contract</p>', unsafe_allow_html=True)
    col10, col11, col12 = st.columns(3)
    with col10:
        contract = st.selectbox(
            "Contract",
            ["Month-To-Month", "One Year", "Two Year"],
            key="contract",
        )
    with col11:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
    with col12:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic Check",
                "Mailed Check",
                "Bank Transfer (Automatic)",
                "Credit Card (Automatic)",
            ],
            key="payment",
        )

    col13, col14, col15 = st.columns(3)
    with col13:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1, key="tenure")
    with col14:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.01, max_value=500.0, value=65.0, step=0.01, key="monthly")
    with col15:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=0.01, key="total")

    # ── Submit ─────────────────────────────────────────────────────────────────
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted:
    payload = {
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           int(tenure),
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     online_back,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_mv,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   float(monthly_charges),
        "TotalCharges":     float(total_charges),
    }

    with st.spinner("Calling the prediction API..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                result = response.json()
                churn_prob = result["churn_probability"]
                prediction = result["prediction"]
                risk_level = result["risk_level"]

                st.markdown("---")
                st.markdown("## 📊 Prediction Result")

                # ── Prediction verdict ─────────────────────────────────────────
                left_col, right_col = st.columns([2, 1])
                with left_col:
                    if prediction == "Churn":
                        st.markdown(
                            f'<div class="churn-box">'
                            f'<h2 style="color:#FF4B4B; margin:0">⚠️ CHURN PREDICTED</h2>'
                            f'<p style="margin:0.5rem 0 0">This customer is <strong>likely to leave</strong>. '
                            f'Consider a retention offer.</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="nochurn-box">'
                            f'<h2 style="color:#21C354; margin:0">✅ NO CHURN PREDICTED</h2>'
                            f'<p style="margin:0.5rem 0 0">This customer is <strong>likely to stay</strong>.</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with right_col:
                    risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.metric("Risk Level", f"{risk_colors.get(risk_level, '')} {risk_level}")
                    st.metric("Model Version", result.get("model_version", "?"))

                # ── Probability bars ───────────────────────────────────────────
                st.markdown("### 📈 Churn Probability")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Churn probability:** `{churn_prob:.1%}`")
                    st.progress(churn_prob)
                with col_b:
                    nc_prob = result["non_churn_probability"]
                    st.markdown(f"**Stay probability:** `{nc_prob:.1%}`")
                    st.progress(nc_prob)

                # ── Top factors ────────────────────────────────────────────────
                if result.get("top_factors"):
                    st.markdown("### 💡 Key Factors")
                    for factor in result["top_factors"]:
                        st.markdown(
                            f'<div class="factor-item">'
                            f'<strong>{factor["feature"]}</strong>: {factor["impact"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            elif response.status_code == 422:
                st.error("❌ Validation error: the API rejected your input.")
                st.json(response.json())
            elif response.status_code == 503:
                st.error("❌ The model is not loaded on the server. Please try again later.")
            else:
                st.error(f"❌ Unexpected error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out after {REQUEST_TIMEOUT}s. Is the API running?")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the backend is running.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
