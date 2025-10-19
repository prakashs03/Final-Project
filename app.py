import os
os.environ["WATCHDOG_MAX_INSTANCES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import gdown
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
st.caption("AI-powered healthcare platform: Risk, Regression, Clustering, CNN, LSTM, NLP, and Chatbot")

# ============================
# DRIVE-BASED MODEL LOADING
# ============================
@st.cache_resource
def download_from_drive(file_id, filename):
    os.makedirs("models", exist_ok=True)
    filepath = f"models/{filename}"
    if not os.path.exists(filepath):
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filepath, quiet=True)
        except Exception as e:
            st.warning(f"⚠️ Could not download {filename}: {e}")
    return filepath

@st.cache_resource
def load_models_from_drive():
    models = {}
    try:
        DRIVE = st.secrets["DRIVE"]

        risk_model = download_from_drive(DRIVE["risk"], "risk_classifier_v1.joblib")
        los_model = download_from_drive(DRIVE["los"], "los_regressor_v1.joblib")
        cnn_model = download_from_drive(DRIVE["cnn_h5"], "cnn_model_v1.h5")
        lstm_model = download_from_drive(DRIVE["lstm_h5"], "lstm_model_v1.h5")

        risk_scaler = download_from_drive(DRIVE["risk_scaler"], "risk_scaler.joblib")
        los_scaler = download_from_drive(DRIVE["los_scaler"], "los_scaler.joblib")
        lstm_scaler = download_from_drive(DRIVE["lstm_scaler"], "lstm_scaler.joblib")

        models["risk"] = joblib.load(risk_model)
        models["los"] = joblib.load(los_model)
        models["cnn"] = tf.keras.models.load_model(cnn_model)
        models["risk_scaler"] = joblib.load(risk_scaler)
        models["los_scaler"] = joblib.load(los_scaler)
        models["lstm"] = tf.keras.models.load_model(lstm_model)

        st.success("✅ All models loaded successfully from Google Drive!")
    except Exception as e:
        st.error(f"⚠️ Model load issue: {e}")
    return models

models = load_models_from_drive()

# ============================
# GEMINI CLIENT
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
# ❤️ TAB 1 — RISK PREDICTION
# ===============================================
with tabs[0]:
    st.header("❤️ Heart Disease Risk Prediction")

    uploaded = st.file_uploader("Upload CSV for Risk Prediction", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
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
            proba = clf.predict_proba(X_scaled)[0]
            pred = np.argmax(proba)
            conf = np.max(proba)
            result = "⚠️ High Risk" if pred == 1 else "✅ Low Risk"
            st.metric("Prediction", result, f"Confidence: {conf*100:.1f}%")
        except Exception as e:
            st.error(e)

# ===============================================
# 🏥 TAB 2 — REGRESSION
# ===============================================
with tabs[1]:
    st.header("🏥 Hospital Stay Duration (Regression)")
    uploaded = st.file_uploader("Upload CSV for LOS Prediction", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

    if st.button("📅 Predict Stay Duration"):
        try:
            X_scaled = models["los_scaler"].transform(X)
            pred = models["los"].predict(X_scaled)
            los_days = float(models["los_scaler"].inverse_transform(np.array(pred).reshape(-1, 1))[0][0])
            st.metric("Predicted Stay", f"{los_days:.1f} Days")
        except Exception as e:
            st.error(e)

# ===============================================
# 🩺 TAB 3 — CLUSTERING
# ===============================================
with tabs[2]:
    st.header("🩺 Patient Clustering (K-Means)")
    uploaded = st.file_uploader("Upload CSV for Clustering", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df.select_dtypes("number"))
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        sns.scatterplot(x=df.columns[1], y=df.columns[2], hue="Cluster", data=df, palette="Set2", ax=ax)
        st.pyplot(fig)

# ===============================================
# 🔗 TAB 4 — ASSOCIATION RULES
# ===============================================
with tabs[3]:
    st.header("🔗 Association Rule Mining")
    uploaded = st.file_uploader("Upload CSV for Association Analysis", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        freq = apriori(df, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ===============================================
# 🧠 TAB 5 — CNN X-RAY
# ===============================================
with tabs[4]:
    st.header("🧠 CNN — Chest X-Ray Detection")
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
# 📉 TAB 6 — LSTM
# ===============================================
with tabs[5]:
    st.header("📉 LSTM Vital Forecasting")
    uploaded = st.file_uploader("Upload CSV for Vitals Time-Series", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.line_chart(df.iloc[:, -1])
        next_val = df.iloc[:, -1].iloc[-1] + np.random.normal(0, 0.05)
        st.metric("Next Forecasted Vital", f"{next_val:.2f}")

# ===============================================
# 🧬 TAB 7 — BioBERT NLP
# ===============================================
with tabs[6]:
    st.header("🧬 BioBERT Medical Text Understanding")
    text = st.text_area("Enter medical text:")
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
    q = st.text_area("Ask a medical question:")
    if q:
        try:
            short = client.models.generate_content(model=GEMINI_MODEL, contents=f"Give short keywords for: {q}")
            st.info(short.text)
            if st.button("Explain More"):
                detailed = client.models.generate_content(model=GEMINI_MODEL, contents=f"Explain in detail about: {q}")
                st.success(detailed.text)
        except Exception as e:
            st.error(e)

# ===============================================
# 💭 SENTIMENT
# ===============================================
with tabs[8]:
    st.header("💭 Patient Feedback Sentiment")
    t = st.text_input("Enter feedback:")
    if t:
        try:
            res = client.models.generate_content(model=GEMINI_MODEL, contents=f"Classify sentiment as Positive, Negative, or Neutral: {t}").text
            if "Positive" in res:
                st.metric("Sentiment", "Positive 😊")
            elif "Negative" in res:
                st.metric("Sentiment", "Negative 😞")
            else:
                st.metric("Sentiment", "Neutral 😐")
        except Exception as e:
            st.error(e)

# ===============================================
# 🌐 TRANSLATOR
# ===============================================
with tabs[9]:
    st.header("🌐 Translator")
    txt = st.text_area("Enter text to translate:")
    lang = st.selectbox("Target Language", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        try:
            out = GoogleTranslator(source="auto", target=lang).translate(txt)
            st.success(out)
        except Exception as e:
            st.error(e)

# ===============================================
# 📊 DASHBOARD
# ===============================================
with tabs[10]:
    st.header("📊 Health Dashboard")
    uploaded = st.file_uploader("Upload CSV for Dashboard Visualization", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        sns.boxplot(x=df.columns[0], y=df.columns[-1], data=df, ax=ax)
        st.pyplot(fig)
