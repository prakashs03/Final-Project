import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
import os

# ---------------------- Gemini Config ----------------------
genai.configure(api_key=st.secrets["GENAI_API_KEY"])
MODEL_NAME = st.secrets.get("GENAI_MODEL", "models/gemini-1.5-flash")

# ---------------------- Helper Functions ----------------------
@st.cache_resource
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load {path}: {e}")
        return None

@st.cache_resource
def safe_load_tf_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.warning(f"⚠️ Failed to load {path}: {e}")
        return None

# ---------------------- Load Models ----------------------
risk_model = safe_load_joblib("models/risk_classifier_v1.joblib")
los_model = safe_load_joblib("models/los_regressor_v1.joblib")
risk_scaler = safe_load_joblib("models/risk_scaler.joblib")
los_scaler = safe_load_joblib("models/los_scaler.joblib")
cnn_model = safe_load_tf_model("models/cnn_model_v1_compat.h5")
lstm_model = safe_load_tf_model("models/lstm_model_v1_compat.h5")
sent_vec = safe_load_joblib("models/sentiment_vectorizer.joblib")
sent_model = safe_load_joblib("models/sentiment_model.joblib")
cluster_model = safe_load_joblib("models/patient_cluster_model.joblib")
cluster_scaler = safe_load_joblib("models/patient_cluster_scaler.joblib")

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="HealthAI", page_icon="⚕️", layout="wide")
st.title("⚕️ HealthAI — Smart Healthcare Assistant")
tabs = st.tabs(["🧬 Risk & LOS", "👥 Clustering", "🧠 LSTM Forecast",
                "🩻 CNN X-ray", "💬 Chatbot", "🌐 Translator", "❤️ Sentiment"])

# ---------------------- 1️⃣ Risk & LOS ----------------------
with tabs[0]:
    st.header("Disease Risk & Length of Stay Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        heart_rate = st.number_input("Heart Rate", 40, 200, 80)
    with col2:
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 40, 120, 80)
    with col3:
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        bmi = st.number_input("BMI", 10.0, 50.0, 24.5)

    if st.button("🔍 Predict"):
        X = np.array([[age, systolic_bp, diastolic_bp, heart_rate, cholesterol, bmi]])
        try:
            risk_val = risk_model.predict(risk_scaler.transform(X))[0]
            los_days = max(1, round(los_model.predict(los_scaler.transform(X))[0]))
            st.success(f"🩸 Disease Risk: **{risk_val}**")
            st.info(f"🏥 Estimated Length of Stay: **{los_days} days**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------- 2️⃣ Clustering ----------------------
with tabs[1]:
    st.header("Patient Clustering")
    try:
        df = pd.read_csv("models/synthetic_health_data.csv")
        feats = ["age", "systolic_bp", "diastolic_bp", "heart_rate",
                 "cholesterol", "blood_sugar", "bmi"]
        if st.button("Cluster Patients"):
            X = cluster_scaler.transform(df[feats])
            df["Cluster"] = cluster_model.predict(X)
            st.dataframe(df.head(10))
    except Exception as e:
        st.warning(f"Clustering failed: {e}")

# ---------------------- 3️⃣ LSTM ----------------------
with tabs[2]:
    st.header("Health Forecast (LSTM)")
    uploaded = st.file_uploader("Upload sequential vital CSV", type=["csv"])
    if uploaded and lstm_model:
        data = pd.read_csv(uploaded)
        try:
            seq = data.values.reshape(1, data.shape[0], data.shape[1])
            preds = lstm_model.predict(seq)
            st.success(f"Predicted future value: {float(preds[0][0]):.2f}")
        except Exception as e:
            st.error(f"LSTM error: {e}")

# ---------------------- 4️⃣ CNN ----------------------
with tabs[3]:
    st.header("X-ray Pneumonia Detection (CNN)")
    img = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])
    if img and cnn_model:
        try:
            image = Image.open(img).resize((128, 128))
            arr = np.expand_dims(np.array(image) / 255.0, axis=0)
            pred = cnn_model.predict(arr)[0][0]
            label = "Pneumonia" if pred > 0.5 else "Normal"
            st.image(image, caption=f"Prediction: {label}")
        except Exception as e:
            st.error(f"CNN error: {e}")

# ---------------------- 5️⃣ Chatbot ----------------------
with tabs[4]:
    st.header("💬 Health Chatbot (Gemini)")
    query = st.text_input("Ask a health-related question:")
    if query:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            reply = model.generate_content(query)
            st.write(reply.text)
        except Exception as e:
            st.error(f"Gemini error: {e}")

# ---------------------- 6️⃣ Translator ----------------------
with tabs[5]:
    st.header("🌐 Translator (Gemini)")
    text = st.text_area("Enter text:")
    lang = st.text_input("Target language (e.g. Tamil, Hindi):")
    if st.button("Translate"):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            res = model.generate_content(f"Translate this to {lang}: {text}")
            st.success(res.text)
        except Exception as e:
            st.error(f"Translation error: {e}")

# ---------------------- 7️⃣ Sentiment ----------------------
with tabs[6]:
    st.header("❤️ Sentiment Analysis")
    review = st.text_area("Enter feedback or review:")
    if st.button("Analyze Sentiment"):
        try:
            vec = sent_vec.transform([review])
            pred = sent_model.predict(vec)[0]
            label = "Positive" if pred == 1 else "Negative"
            st.success(f"Sentiment: {label}")
        except Exception as e:
            st.error(f"Sentiment error: {e}")

st.caption("🧠 Powered by TensorFlow, scikit-learn, and Gemini AI | © 2025 HealthAI")
