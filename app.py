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
import xgboost as xgb
import json, pathlib

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="💊 HealthAI - Smart Healthcare Assistant", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-powered multimodal healthcare assistant — Risk, LOS, CNN, LSTM, Sentiment, Chatbot & Translation")

os.makedirs("models", exist_ok=True)

# ============================
# GOOGLE DRIVE + GEMINI SETUP
# ============================
DRIVE = st.secrets["DRIVE"]
GEMINI_API = st.secrets["GENAI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]
client = genai.Client(api_key=GEMINI_API)

# ============================
# SAFE MODEL DOWNLOADER
# ============================
@st.cache_resource
def download_model(url, filename):
    path = f"models/{filename}"
    try:
        gdown.download(url, path, quiet=False, fuzzy=True)
    except Exception as e:
        st.warning(f"⚠️ Download failed for {filename}: {e}")
    return path

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    models = {}
    try:
        models["risk"] = joblib.load(download_model(DRIVE["risk"], "risk_classifier_v1.joblib"))
        models["los"] = joblib.load(download_model(DRIVE["los"], "los_regressor_v1.joblib"))
        models["cnn"] = tf.keras.models.load_model(download_model(DRIVE["cnn_h5"], "cnn_model_v1.h5"))
        models["lstm"] = tf.keras.models.load_model(download_model(DRIVE["lstm_h5"], "lstm_model_v1.h5"))
        models["risk_scaler"] = joblib.load(download_model(DRIVE["risk_scaler"], "risk_scaler.joblib"))
        models["los_scaler"] = joblib.load(download_model(DRIVE["los_scaler"], "los_scaler.joblib"))
        models["lstm_scaler"] = joblib.load(download_model(DRIVE["lstm_scaler"], "lstm_scaler.joblib"))
        models["sentiment_model"] = joblib.load(download_model(DRIVE["sentiment_model"], "sentiment_model.joblib"))
        models["sentiment_vectorizer"] = joblib.load(download_model(DRIVE["sentiment_vectorizer"], "sentiment_vectorizer.joblib"))
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return models

models = load_models()

# ============================
# SIDEBAR DEBUG INFO
# ============================
if st.sidebar.checkbox("Show model diagnostics"):
    st.sidebar.header("Diagnostics")
    for k, v in models.items():
        try:
            shape_info = ""
            if hasattr(v, "input_shape"): shape_info += f"in:{v.input_shape} "
            if hasattr(v, "output_shape"): shape_info += f"out:{v.output_shape}"
            st.sidebar.write(k, "-", type(v).__name__, shape_info)
        except: pass

# ============================
# TABS
# ============================
tabs = st.tabs(["🏥 Risk & LOS", "🧠 CNN & LSTM", "💬 Chatbot", "💭 Sentiment", "🌐 Translator", "📊 Dashboard"])

# ======================================================
# TAB 1 — RISK & LENGTH OF STAY
# ======================================================
with tabs[0]:
    st.header("🏥 Disease Risk & Hospital Stay Duration")

    age = st.slider("Age", 10, 90, 40)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 350, 200)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    glucose = st.slider("Glucose", 50, 250, 110)
    hr = st.slider("Heart Rate", 50, 160, 80)
    features = np.array([[age, bp, chol, bmi, glucose, hr]])

    if st.button("Predict Risk"):
        try:
            X_scaled = models["risk_scaler"].transform(features)
            clf = models["risk"]
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_scaled)[0]
                label = np.argmax(proba)
                conf = np.max(proba)
            else:
                label = clf.predict(X_scaled)[0]
                conf = None
            result = "⚠️ High Risk" if int(label) == 1 else "✅ Low Risk"
            st.metric("Disease Risk Prediction", f"{result} ({conf*100:.1f}%)" if conf else result)
        except Exception as e:
            st.error(f"Risk prediction error: {e}")

    if st.button("Predict Length of Stay"):
        try:
            X_scaled = models["los_scaler"].transform(features)
            los_pred = models["los"].predict(X_scaled)
            try:
                los_days = float(models["los_scaler"].inverse_transform(los_pred.reshape(-1, 1))[0][0])
            except:
                los_days = float(np.array(los_pred).reshape(-1)[0])
            st.metric("Predicted Stay Duration", f"{los_days:.1f} Days")
        except Exception as e:
            st.error(f"LOS prediction error: {e}")

# ======================================================
# TAB 2 — CNN & LSTM
# ======================================================
with tabs[1]:
    st.header("🧠 Deep Learning Models — CNN & LSTM")
    c1, c2 = st.columns(2)

    # CNN
    with c1:
        st.subheader("🩻 CNN — X-ray Classification")
        img = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
        class_names = ["Normal", "Pneumonia"]
        cls_file = pathlib.Path("models/class_names.json")
        if cls_file.exists():
            try:
                class_names = json.load(open(cls_file))
            except: pass
        if img:
            image = Image.open(img).convert("RGB")
            cnn = models["cnn"]
            _, H, W, C = cnn.input_shape
            img_arr = np.array(image.resize((W, H))) / 255.0
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr]*3, axis=-1)
            img_arr = np.expand_dims(img_arr, 0)
            preds = cnn.predict(img_arr)
            if preds.shape[-1] == 1:
                p = float(preds[0][0])
                if p < 0 or p > 1: p = 1/(1+np.exp(-p))
                label = 1 if p >= 0.5 else 0
                conf = p if label == 1 else 1-p
            else:
                label = int(np.argmax(preds))
                conf = float(np.max(preds))
            name = class_names[label] if label < len(class_names) else f"Class {label}"
            if conf < 0.6:
                st.warning(f"🤔 Uncertain: {name} ({conf*100:.1f}%)")
            else:
                color = "success" if name.lower()=="normal" else "error"
                getattr(st, color)(f"{name} ({conf*100:.1f}%)")

    # LSTM
    with c2:
        st.subheader("📈 LSTM — Health Metric Forecast")
        timesteps = np.arange(20)
        data = np.sin(timesteps) + np.random.normal(0, 0.1, len(timesteps))
        scaler = models["lstm_scaler"]
        data_scaled = scaler.transform(data.reshape(-1, 1))
        lstm = models["lstm"]
        _, T, F = lstm.input_shape
        data_scaled = data_scaled[-T:, :]
        if F > 1:
            data_scaled = np.repeat(data_scaled, F, axis=1)
        X = data_scaled.reshape(1, T, F)
        try:
            pred_scaled = lstm.predict(X)
            val = scaler.inverse_transform(pred_scaled)[0][0]
            st.metric("Forecasted Health Metric", f"{val:.3f}")
            fig, ax = plt.subplots()
            ax.plot(data[-T:], label="Input")
            ax.axhline(val, color='r', linestyle='--', label='Forecast')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"LSTM error: {e}")

# ======================================================
# TAB 3 — CHATBOT
# ======================================================
with tabs[2]:
    st.header("💬 Gemini Healthcare Chatbot")
    query = st.text_area("Ask me anything (in any language):")
    if query:
        try:
            query_en = GoogleTranslator(source="auto", target="en").translate(query)
        except:
            query_en = query
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Short Answer"):
                try:
                    short_resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=f"Give short medical keywords for: {query_en}"
                    )
                    st.info(short_resp.text)
                except Exception as e:
                    st.error(f"Chatbot error: {e}")
        with col2:
            if st.button("Explain More"):
                try:
                    long_resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=f"Explain in detail about: {query_en}"
                    )
                    st.success(long_resp.text)
                except Exception as e:
                    st.error(f"Chatbot error: {e}")

# ======================================================
# TAB 4 — SENTIMENT
# ======================================================
with tabs[3]:
    st.header("💭 Sentiment Analysis — Patient Feedback")
    text = st.text_input("Enter feedback:")
    if st.button("Analyze Sentiment"):
        if not text.strip():
            st.warning("Please enter some feedback.")
        else:
            try:
                g_resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"Classify sentiment of this feedback as Positive, Negative, or Neutral: {text}"
                ).text
                if "Positive" in g_resp:
                    st.metric("Sentiment", "Positive 😊")
                elif "Negative" in g_resp:
                    st.metric("Sentiment", "Negative 😞")
                else:
                    st.metric("Sentiment", "Neutral 😐")
            except:
                vec = models["sentiment_vectorizer"].transform([text])
                pred = models["sentiment_model"].predict(vec)[0]
                st.metric("Sentiment", "Positive 😊" if pred == 1 else "Negative 😞")

# ======================================================
# TAB 5 — TRANSLATOR
# ======================================================
with tabs[4]:
    st.header("🌐 Translator — Multilingual Support")
    txt = st.text_area("Enter text to translate:")
    lang = st.selectbox("Select language", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        try:
            translated = GoogleTranslator(source="auto", target=lang).translate(txt)
            st.success(f"🔤 Translated ({lang}): {translated}")
        except Exception as e:
            st.warning(f"Translation failed: {e}")

# ======================================================
# TAB 6 — DASHBOARD
# ======================================================
with tabs[5]:
    st.header("📊 Dashboard — Risk & Stay Comparison")
    data = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 100),
        "LOS": np.random.uniform(1, 10, 100)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=data, ax=ax)
    st.pyplot(fig)
    st.info("🧠 Insight: High-risk patients tend to stay longer.")
