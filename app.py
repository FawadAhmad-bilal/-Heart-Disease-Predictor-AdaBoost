import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0f0c29, #1a1a40, #24243e);
}

.heart-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}

.heart-icon { font-size: 72px; animation: pulse 1.5s infinite; }

@keyframes pulse {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}

.app-caption {
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    margin-top: 4px;
}

.section-head {
    color: #e879f9;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.6rem;
}

/* inputs */
input[type="number"],
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #fff !important;
}

/* predict button */
.stButton > button {
    background: linear-gradient(135deg, #e879f9, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
}

/* metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
}

/* hide default UI */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('adaboost_heart.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found!")
    st.stop()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="heart-header">
    <div class="heart-icon">🫀</div>
    <div class="app-title">Heart Disease Predictor</div>
    <div class="app-caption">AdaBoost ML Model · 81.5% Accuracy · UCI Dataset</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Patient Info ──────────────────────────────────────────────
st.markdown('<div class="section-head">👤 Patient Info</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])

# ── Cardiac Measurements ──────────────────────────────────────
st.markdown('<div class="section-head">💓 Cardiac Measurements</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    thalch   = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=140)
with col4:
    chol    = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=200)
    oldpeak = st.number_input("Oldpeak — ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# ── Clinical Details ──────────────────────────────────────────
st.markdown('<div class="section-head">🩺 Clinical Details</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    cp      = st.selectbox("Chest Pain Type", ["asymptomatic", "typical angina", "atypical angina", "non-anginal"])
    restecg = st.selectbox("Resting ECG Result", ["normal", "lv hypertrophy", "st-t abnormality"])
with col6:
    fbs   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍  Run Prediction", use_container_width=True):

    sex_enc     = 1 if sex == "Male" else 0
    fbs_enc     = 1 if fbs == "True" else 0
    exang_enc   = 1 if exang == "True" else 0
    restecg_enc = {"normal": 1, "lv hypertrophy": 0, "st-t abnormality": 2}[restecg]
    cp_atypical   = 1 if cp == "atypical angina" else 0
    cp_nonanginal = 1 if cp == "non-anginal" else 0
    cp_typical    = 1 if cp == "typical angina" else 0

    input_data = pd.DataFrame([{
        'age':                age,
        'sex':                sex_enc,
        'trestbps':           trestbps,
        'chol':               chol,
        'fbs':                fbs_enc,
        'restecg':            restecg_enc,
        'thalch':             thalch,
        'exang':              exang_enc,
        'oldpeak':            oldpeak,
        'cp_atypical angina': cp_atypical,
        'cp_non-anginal':     cp_nonanginal,
        'cp_typical angina':  cp_typical,
    }])

    result = model.predict(input_data)[0]
    proba  = model.predict_proba(input_data)[0]

    st.divider()

    if result == 1:
        st.error("### 🚨 Heart Disease Detected!")
        st.warning("Please consult a cardiologist immediately.")
    else:
        st.success("### ✅ No Heart Disease Detected!")
        st.info("Patient appears healthy. Maintain a good lifestyle!")

    st.subheader("📊 Prediction Confidence")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.metric("No Disease", f"{proba[0]*100:.1f}%")
    with col8:
        st.metric("Disease", f"{proba[1]*100:.1f}%")
    with col9:
        st.metric("Model Accuracy", "81.5%")

    with st.expander("📋 View Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Age", "Sex", "Chest Pain", "Blood Pressure",
                        "Cholesterol", "Fasting Blood Sugar", "ECG",
                        "Max Heart Rate", "Exercise Angina", "Oldpeak"],
            "Value":   [age, sex, cp, f"{trestbps} mmHg",
                        f"{chol} mg/dl", fbs, restecg,
                        thalch, exang, oldpeak]
        })
        st.table(summary)

st.divider()
st.caption("⚠️ Educational purposes only — not a medical diagnosis tool | Fawad Ahmad Bilal · BSAI · University of Haripur")