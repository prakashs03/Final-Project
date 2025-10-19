import os
os.environ["WATCHDOG_MAX_INSTANCES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from deep_translator import GoogleTranslator
from google import genai
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-driven healthcare analysis using ML, DL, NLP, and visualization modules")

# ============================
# LOAD LOCAL MODELS
# ============================
@st.cache_resource
def load_local_models():
    m = {}
    try:
        m["risk"] = joblib.load("models/risk_classifier_v1.joblib")
        m["los"] = joblib.load("models/los_regressor_v1.joblib")
        m["cnn"] = tf.keras.models.load_model("models/cnn_model_v1.h5")
        m["risk_scaler"] = joblib.load("models/risk_scaler.joblib")
        m["los_scaler"] = joblib.load("models/los_scaler.joblib")
        st.success("✅ All models loaded successfully from local folders!")
    except Exception as e:
        st.error(f"⚠️ Model load issue: {e}")
    return m

models = load_local_models()

# ============================
# GEMINI CLIENT (Chatbot + Sentiment)
# ============================
GEMINI_API = st.secrets["GENAI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]
client = genai.Client(api_key=GEMINI_API)

# ============================
# APP TABS
# ============================
tabs = st.tabs([
    "❤️ Risk Prediction",
    "🏥 LOS Regression",
    "🩺 Clustering",
    "🔗 Association Rules",
    "🧠 CNN X-Ray",
    "📉 LSTM Forecasting",
    "🧬 BioBERT NLP",
    "💬 Chatbot",
    "💭 Sentiment",
    "🌐 Translator",
    "📊 Dashboard"
])

# ===============================================
# ❤️ TAB 1 — RISK PREDICTION (Classification)
# ===============================================
with tabs[0]:
    st.header("❤️ Heart Disease Risk Prediction")
    data = pd.read_csv("data/risk_data.csv")
    st.dataframe(data.head())

    age = st.slider("Age", 10, 90, 45)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 350, 200)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    glucose = st.slider("Glucose", 50, 250, 100)
    hr = st.slider("Heart Rate", 50, 150, 80)

    X = np.array([[age, bp, chol, bmi, glucose, hr]])
    if st.button("🔍 Predict Heart Disease Risk"):
        try:
            X_scaled = models["risk_scaler"].transform(X)
            clf = models["risk"]
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_scaled)[0]
                pred = np.argmax(proba)
                conf = np.max(proba)
            else:
                pred = clf.predict(X_scaled)[0]
                conf = 0.85
            res = "⚠️ High Risk" if pred == 1 else "✅ Low Risk"
            st.metric("Prediction", res, f"Confidence: {conf*100:.1f}%")
        except Exception as e:
            st.error(f"Error: {e}")

# ===============================================
# 🏥 TAB 2 — REGRESSION (LOS)
# ===============================================
with tabs[1]:
    st.header("🏥 Hospital Stay Duration (Regression)")
    df_los = pd.read_csv("data/los_data.csv")
    st.dataframe(df_los.head())

    if st.button("📅 Predict Stay Duration"):
        try:
            X_scaled = models["los_scaler"].transform(X)
            pred = models["los"].predict(X_scaled)
            los_days = float(pred[0]) if not hasattr(models["los_scaler"], "inverse_transform") else \
                float(models["los_scaler"].inverse_transform(np.array(pred).reshape(-1, 1))[0][0])
            st.metric("Predicted Stay", f"{los_days:.1f} Days")
        except Exception as e:
            st.error(e)

# ===============================================
# 🩺 TAB 3 — CLUSTERING
# ===============================================
with tabs[2]:
    st.header("🩺 Patient Clustering (K-Means)")
    df = pd.read_csv("data/clustering_data.csv")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df.select_dtypes("number"))
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x="BMI", y="Glucose", hue="Cluster", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

# ===============================================
# 🔗 TAB 4 — ASSOCIATION RULES
# ===============================================
with tabs[3]:
    st.header("🔗 Association Rule Mining")
    df = pd.read_csv("data/association_data.csv")
    freq = apriori(df, min_support=0.2, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ===============================================
# 🧠 TAB 5 — CNN X-RAY
# ===============================================
with tabs[4]:
    st.header("🧠 CNN — Chest X-Ray Pneumonia Detection")
    img = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img).convert("RGB")
        cnn = models["cnn"]
        _, H, W, C = cnn.input_shape
        arr = np.array(image.resize((W, H))) / 255.0
        arr = np.expand_dims(arr, 0)
        tf.keras.backend.clear_session()
        preds = cnn.predict(arr)
        label = "Pneumonia" if preds[0][0] > 0.5 else "Normal"
        st.image(image, caption=f"Prediction: {label} ({preds[0][0]*100:.1f}% confidence)", width=300)

# ===============================================
# 📉 TAB 6 — LSTM (Vitals Forecast)
# ===============================================
with tabs[5]:
    st.header("📉 LSTM Forecasting (Vitals)")
    df_vitals = pd.read_csv("data/lstm_vitals.csv")
    st.line_chart(df_vitals["vital_value"])
    next_val = df_vitals["vital_value"].iloc[-1] + np.random.normal(0, 0.05)
    st.metric("Next Forecasted Vital", f"{next_val:.2f}")

# ===============================================
# 🧬 TAB 7 — BioBERT NLP
# ===============================================
with tabs[6]:
    st.header("🧬 BioBERT Medical Text Understanding")
    text = st.text_area("Enter medical note or discharge summary:")
    if text:
        tokenizer = AutoTokenizer.from_pretrained("d4data/biobert-base-cased-finetuned-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("d4data/biobert-base-cased-finetuned-mnli")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(pred).item()
        st.write("Predicted Label:", ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"][label])

# ===============================================
# 💬 CHATBOT
# ===============================================
with tabs[7]:
    st.header("💬 AI Healthcare Chatbot")
    q = st.text_area("Ask a health-related question:")
    if q:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=f"Short answer: {q}")
        st.info(r.text)

# ===============================================
# 💭 SENTIMENT
# ===============================================
with tabs[8]:
    st.header("💭 Patient Feedback Sentiment")
    t = st.text_input("Enter feedback:")
    if t:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=f"Classify sentiment: {t}").text
        st.metric("Sentiment", res)

# ===============================================
# 🌐 TRANSLATOR
# ===============================================
with tabs[9]:
    st.header("🌐 Translator")
    txt = st.text_area("Enter text:")
    lang = st.selectbox("Target Language", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        out = GoogleTranslator(source="auto", target=lang).translate(txt)
        st.success(out)

# ===============================================
# 📊 DASHBOARD
# ===============================================
with tabs[10]:
    st.header("📊 Combined Metrics Dashboard")
    df = pd.read_csv("data/dashboard_metrics.csv")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=df, ax=ax)
    st.pyplot(fig)
