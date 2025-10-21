# ===============================
# HealthAI – Smart Healthcare Assistant
# Final Streamlit App (GitHub Model Version)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
import os

# -------------------------------
# 🔐 GEMINI API CONFIG
# -------------------------------
genai.configure(api_key=st.secrets["GENAI_API_KEY"])
MODEL_NAME = st.secrets["GENAI_MODEL"]

# -------------------------------
# 📦 LOAD MODELS SAFELY
# -------------------------------
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load {path}: {e}")
        return None

def safe_load_keras(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.warning(f"⚠️ Failed to load {path}: {e}")
        return None

@st.cache_resource
def load_models():
    models = {
        "risk_model": safe_load_joblib("models/risk_classifier_v1.joblib"),
        "los_model": safe_load_joblib("models/los_regressor_v1.joblib"),
        "los_scaler": safe_load_joblib("models/los_scaler.joblib"),
        "risk_scaler": safe_load_joblib("models/risk_scaler.joblib"),
        "lstm_model": safe_load_keras("models/lstm_model_v1_compat.h5"),
        "cnn_model": safe_load_keras("models/cnn_model_v1_compat.h5"),
        "sentiment_model": safe_load_joblib("models/sentiment_model.joblib"),
        "sentiment_vectorizer": safe_load_joblib("models/sentiment_vectorizer.joblib"),
        "cluster_model": safe_load_joblib("models/patient_cluster_model.joblib"),
        "cluster_scaler": safe_load_joblib("models/patient_cluster_scaler.joblib"),
    }
    return models

models = load_models()

# -------------------------------
# 🏥 STREAMLIT UI SETUP
# -------------------------------
st.set_page_config(page_title="HealthAI", layout="wide", page_icon="⚕️")
st.title("⚕️ HealthAI — Smart Healthcare Assistant")
st.markdown("Empowered by AI for health prediction, insights, and interaction.")

tabs = st.tabs([
    "🧬 Disease Risk & Stay Prediction",
    "👥 Patient Clustering",
    "🧠 LSTM Health Forecast",
    "🩻 CNN (X-ray Diagnosis)",
    "💬 Health Chatbot (Gemini)",
    "🌐 Translator (Gemini)",
    "❤️ Sentiment Analysis (Gemini)"
])

# -------------------------------
# 🧬 TAB 1: RISK & LOS
# -------------------------------
with tabs[0]:
    st.header("Disease Risk & Length of Stay Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        heart_rate = st.number_input("Heart Rate", 50, 180, 80)
    with col2:
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 40, 120, 80)
    with col3:
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        bmi = st.number_input("BMI", 10.0, 50.0, 24.5)

    if st.button("🔍 Predict Disease Risk & Stay"):
        try:
            X = np.array([[age, systolic_bp, diastolic_bp, heart_rate, cholesterol, bmi]])
            X_scaled_risk = models["risk_scaler"].transform(X)
            risk_pred = models["risk_model"].predict(X_scaled_risk)[0]

            X_scaled_los = models["los_scaler"].transform(X)
            los_pred = models["los_model"].predict(X_scaled_los)[0]
            los_days = max(1, round(los_pred))

            st.success(f"🩸 Predicted Disease Risk: **{risk_pred}**")
            st.info(f"🏥 Estimated Length of Stay: **{los_days} days**")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

# -------------------------------
# 👥 TAB 2: PATIENT CLUSTERING
# -------------------------------
with tabs[1]:
    st.header("Patient Segmentation (Clustering)")
    st.caption("Groups patients into Low / Medium / High risk clusters based on health data.")
    try:
        df = pd.read_csv("models/synthetic_health_data.csv")
        if st.button("🧩 Show Cluster Insights"):
            scaled = models["cluster_scaler"].transform(
                df[["age", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol", "blood_sugar", "bmi"]]
            )
            clusters = models["cluster_model"].predict(scaled)
            df["Cluster"] = clusters
            st.dataframe(df.head(10))
            st.success("Clustered into: 0 = Low risk, 1 = Medium, 2 = High.")
    except Exception as e:
        st.warning(f"⚠️ Could not cluster: {e}")

# -------------------------------
# 🧠 TAB 3: LSTM Forecast
# -------------------------------
with tabs[2]:
    st.header("Health Forecast (LSTM Model)")
    st.caption("Forecasts patient vitals using time-series LSTM model.")
    if models["lstm_model"]:
        uploaded = st.file_uploader("Upload CSV with Time Series (Heart Rate, BP, etc.)", type=["csv"])
        if uploaded:
            try:
                data = pd.read_csv(uploaded)
                seq = data.values.reshape(1, data.shape[0], data.shape[1])
                preds = models["lstm_model"].predict(seq)
                st.success(f"🧩 Forecasted Value: {float(preds[0][0]):.2f}")
            except Exception as e:
                st.error(f"⚠️ LSTM error: {e}")
    else:
        st.info("LSTM model unavailable.")

# -------------------------------
# 🩻 TAB 4: CNN IMAGE DIAGNOSIS
# -------------------------------
with tabs[3]:
    st.header("CNN – X-ray Pneumonia Detection")
    uploaded_img = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])
    if uploaded_img and models["cnn_model"]:
        try:
            img = Image.open(uploaded_img).resize((128, 128))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = models["cnn_model"].predict(arr)
            label = "Pneumonia" if preds[0][0] > 0.5 else "Normal"
            st.image(img, caption=f"Prediction: {label}")
            st.success(f"🩻 The X-ray is classified as **{label}**")
        except Exception as e:
            st.error(f"⚠️ CNN prediction error: {e}")
    else:
        st.info("Upload image to analyze.")

# -------------------------------
# 💬 TAB 5: CHATBOT
# -------------------------------
with tabs[4]:
    st.header("💬 Health Chatbot (Gemini)")
    user_input = st.text_input("Ask a health-related question:")
    if user_input:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(f"Answer briefly about: {user_input}")
            summary = response.text
            st.write(summary)

            if st.button("📘 Explain in detail"):
                detailed = model.generate_content(f"Explain in detail about: {user_input}")
                st.write(detailed.text)
        except Exception as e:
            st.error(f"⚠️ Chatbot error: {e}")

# -------------------------------
# 🌐 TAB 6: TRANSLATOR
# -------------------------------
with tabs[5]:
    st.header("🌐 Translator (Gemini)")
    text = st.text_area("Enter text to translate:")
    lang = st.text_input("Enter target language (e.g., Tamil, Hindi, French):")
    if st.button("Translate"):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            output = model.generate_content(f"Translate this to {lang}: {text}")
            st.success(output.text)
        except Exception as e:
            st.error(f"⚠️ Translation error: {e}")

# -------------------------------
# ❤️ TAB 7: SENTIMENT
# -------------------------------
with tabs[6]:
    st.header("❤️ Sentiment Analysis (Gemini)")
    text = st.text_area("Enter a health review or feedback:")
    if st.button("Analyze Sentiment"):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            output = model.generate_content(f"Classify sentiment (positive/negative/neutral): {text}")
            st.success(f"Sentiment: {output.text.strip()}")
        except Exception as e:
            st.error(f"⚠️ Sentiment analysis error: {e}")

st.markdown("---")
st.caption("🧠 Powered by TensorFlow, scikit-learn, and Gemini AI | © 2025 HealthAI")
