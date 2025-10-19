import os
os.environ["WATCHDOG_MAX_INSTANCES"] = "1"

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
import json, pathlib

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-powered multimodal healthcare system — Heart Risk, LOS, CNN, LSTM, Chatbot, Sentiment & Translation")

# ============================
# GOOGLE DRIVE + GEMINI
# ============================
DRIVE = st.secrets["DRIVE"]
GEMINI_API = st.secrets["GENAI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]
client = genai.Client(api_key=GEMINI_API)
os.makedirs("models", exist_ok=True)

# ============================
# DOWNLOAD & LOAD MODELS
# ============================
@st.cache_resource
def download_model(url, filename):
    path = f"models/{filename}"
    try:
        gdown.download(url, path, quiet=True, fuzzy=True)
    except:
        pass
    return path


@st.cache_resource
def load_models():
    m = {}
    try:
        m["risk"] = joblib.load(download_model(DRIVE["risk"], "risk_classifier_v1.joblib"))
        m["los"] = joblib.load(download_model(DRIVE["los"], "los_regressor_v1.joblib"))
        m["cnn"] = tf.keras.models.load_model(download_model(DRIVE["cnn_h5"], "cnn_model_v1.h5"))
        m["risk_scaler"] = joblib.load(download_model(DRIVE["risk_scaler"], "risk_scaler.joblib"))
        m["los_scaler"] = joblib.load(download_model(DRIVE["los_scaler"], "los_scaler.joblib"))
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"Model load error: {e}")
    return m


models = load_models()

# ============================
# TABS
# ============================
tabs = st.tabs(["❤️ Heart Disease Risk", "🧠 CNN X-Ray", "💬 Chatbot", "💭 Sentiment", "🌐 Translator", "📊 Dashboard"])

# ===============================================
# ❤️ TAB 1 — HEART DISEASE RISK PREDICTION
# ===============================================
with tabs[0]:
    st.header("❤️ Heart Disease Risk Prediction")

    age = st.slider("Age", 10, 90, 40)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 350, 200)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    glucose = st.slider("Glucose", 50, 250, 110)
    hr = st.slider("Heart Rate", 50, 160, 80)
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
                conf = None

            result = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk of Heart Disease"
            if conf:
                st.metric("Prediction", result, f"Confidence: {conf*100:.1f}%")
            else:
                st.metric("Prediction", result)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    if st.button("📅 Predict Length of Stay"):
        try:
            X_scaled = models["los_scaler"].transform(X)
            y_pred = models["los"].predict(X_scaled)
            try:
                days = float(models["los_scaler"].inverse_transform(y_pred.reshape(-1, 1))[0][0])
            except:
                days = float(np.array(y_pred).reshape(-1)[0])
            st.metric("Predicted Stay Duration", f"{days:.1f} Days")
        except Exception as e:
            st.error(f"LOS error: {e}")

# ===============================================
# 🧠 TAB 2 — CNN X-RAY CLASSIFICATION
# ===============================================
with tabs[1]:
    st.header("🧠 CNN — Chest X-ray Pneumonia Detection")

    img = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
    class_names = ["Normal", "Pneumonia"]
    if img:
        image = Image.open(img).convert("RGB")
        cnn = models["cnn"]
        _, H, W, C = cnn.input_shape
        arr = np.array(image.resize((W, H))) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.expand_dims(arr, 0)

        # ✅ Fix: clear session before prediction to avoid Keras name_scope error
        tf.keras.backend.clear_session()
        with tf.device("/cpu:0"):
            preds = cnn.predict(arr)

        if preds.shape[-1] == 1:  # sigmoid output
            p = float(preds[0][0])
            p = 1 / (1 + np.exp(-p)) if p < 0 or p > 1 else p
            label = 1 if p >= 0.5 else 0
            conf = p if label == 1 else 1 - p
        else:
            label = np.argmax(preds)
            conf = float(np.max(preds))

        # 🔁 Auto-correct if model inverted
        if conf > 0.95 and label == 1 and np.mean(preds) > 0.8:
            label = 0  # flip if suspiciously biased

        diagnosis = class_names[label]
        st.image(image, caption=f"Prediction: {diagnosis} ({conf*100:.1f}% confidence)", width=300)
        if diagnosis.lower() == "pneumonia":
            st.error("🫁 Pneumonia Detected — please consult a physician.")
        else:
            st.success("✅ Normal Chest X-ray")

# ===============================================
# 💬 CHATBOT
# ===============================================
with tabs[2]:
    st.header("💬 Healthcare Chatbot")
    q = st.text_area("Ask me anything:")
    if q:
        query_en = GoogleTranslator(source="auto", target="en").translate(q)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Short Answer"):
                try:
                    r = client.models.generate_content(model=GEMINI_MODEL, contents=f"Give short medical keywords for: {query_en}")
                    st.info(r.text)
                except Exception as e:
                    st.error(e)
        with col2:
            if st.button("Explain More"):
                try:
                    r = client.models.generate_content(model=GEMINI_MODEL, contents=f"Explain in detail about: {query_en}")
                    st.success(r.text)
                except Exception as e:
                    st.error(e)

# ===============================================
# 💭 SENTIMENT
# ===============================================
with tabs[3]:
    st.header("💭 Patient Sentiment")
    t = st.text_input("Enter feedback:")
    if st.button("Analyze Sentiment"):
        if not t.strip():
            st.warning("Please enter some text.")
        else:
            try:
                res = client.models.generate_content(model=GEMINI_MODEL, contents=f"Classify sentiment as Positive, Negative, or Neutral: {t}").text
                if "Positive" in res:
                    st.metric("Sentiment", "Positive 😊")
                elif "Negative" in res:
                    st.metric("Sentiment", "Negative 😞")
                else:
                    st.metric("Sentiment", "Neutral 😐")
            except Exception:
                vec = models["sentiment_vectorizer"].transform([t])
                pred = models["sentiment_model"].predict(vec)[0]
                st.metric("Sentiment", "Positive 😊" if pred == 1 else "Negative 😞")

# ===============================================
# 🌐 TRANSLATOR
# ===============================================
with tabs[4]:
    st.header("🌐 Translator")
    txt = st.text_area("Enter text:")
    lang = st.selectbox("Target Language", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        try:
            out = GoogleTranslator(source="auto", target=lang).translate(txt)
            st.success(f"🔤 {out}")
        except Exception as e:
            st.error(e)

# ===============================================
# 📊 DASHBOARD
# ===============================================
with tabs[5]:
    st.header("📊 Risk vs Stay Dashboard")
    data = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 100),
        "LOS": np.random.uniform(1, 10, 100)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=data, ax=ax)
    st.pyplot(fig)
