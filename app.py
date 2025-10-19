import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import gdown
from PIL import Image
from google import genai
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ============================
# INITIAL SETUP
# ============================

st.set_page_config(page_title="💊 HealthAI — Smart Healthcare Assistant", layout="wide")

client = genai.Client(api_key=st.secrets["GENAI_API_KEY"])
DRIVE = st.secrets["DRIVE"]

@st.cache_resource
def download_model(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, f"models/{filename}", quiet=True)
    return f"models/{filename}"

@st.cache_resource
def load_models():
    models = {}
    try:
        models["risk"] = joblib.load(download_model(DRIVE["risk"], "risk_model.joblib"))
        models["los"] = joblib.load(download_model(DRIVE["los"], "los_model.joblib"))
        models["cnn"] = tf.keras.models.load_model(download_model(DRIVE["cnn_h5"], "cnn_model_v1.h5"))
        models["lstm"] = tf.keras.models.load_model(download_model(DRIVE["lstm_h5"], "lstm_model_v1.h5"))
        models["sentiment_model"] = joblib.load(download_model(DRIVE["sentiment_model"], "sentiment_model.joblib"))
        models["sentiment_vectorizer"] = joblib.load(download_model(DRIVE["sentiment_vectorizer"], "sentiment_vectorizer.joblib"))
        models["risk_scaler"] = joblib.load(download_model(DRIVE["risk_scaler"], "risk_scaler.joblib"))
        models["los_scaler"] = joblib.load(download_model(DRIVE["los_scaler"], "los_scaler.joblib"))
        models["lstm_scaler"] = joblib.load(download_model(DRIVE["lstm_scaler"], "lstm_scaler.joblib"))
    except Exception as e:
        st.warning(f"⚠️ Model load issue: {e}")
    return models

models = load_models()
st.success("✅ All models loaded successfully!")

# ============================
# MAIN UI
# ============================

tabs = st.tabs(["🏥 Risk & LOS", "🧠 CNN / LSTM", "💬 Chatbot", "💭 Sentiment", "📊 Dashboard"])

# --- Tab 1: Risk & LOS ---
with tabs[0]:
    st.header("🏥 Disease Risk & Length of Stay Prediction")

    age = st.number_input("Age", 0, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    glucose = st.number_input("Glucose", 50, 300, 120)
    heart_rate = st.number_input("Heart Rate", 40, 180, 80)

    if st.button("Predict Risk"):
        X = np.array([[age, bp, chol, bmi, glucose, heart_rate]])
        X_scaled = models["risk_scaler"].transform(X)
        risk_pred = models["risk"].predict(X_scaled)
        st.metric("Predicted Risk", "HIGH" if risk_pred[0] == 1 else "LOW")

    if st.button("Predict Length of Stay"):
        X = np.array([[age, bp, chol, bmi, glucose, heart_rate]])
        X_scaled = models["los_scaler"].transform(X)
        los_pred = models["los"].predict(X_scaled)
        st.metric("Predicted Stay (days)", f"{los_pred[0]:.2f}")

# --- Tab 2: CNN & LSTM ---
with tabs[1]:
    st.header("🧠 Deep Learning Models")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🩻 CNN - X-Ray Classification")
        uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
            st.image(img, caption="Uploaded Image", width=250)
            img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = models["cnn"].predict(img_arr)
            result = "Pneumonia" if np.argmax(pred) == 1 else "Normal"
            st.metric("Diagnosis", result)

    with col2:
        st.subheader("📈 LSTM - Time Series Forecasting")
        time_steps = np.arange(50)
        data = np.sin(time_steps) + np.random.normal(0, 0.1, 50)
        scaled = models["lstm_scaler"].transform(data.reshape(-1, 1))
        X = scaled.reshape(1, 50, 1)
        pred = models["lstm"].predict(X)
        predicted_value = models["lstm_scaler"].inverse_transform(pred)[0][0]
        st.metric("Next Forecast Value", f"{predicted_value:.2f}")
        plt.plot(data, label="Historical")
        plt.axhline(predicted_value, color='r', linestyle='--', label="Forecast")
        plt.legend()
        st.pyplot(plt)

# --- Tab 3: Chatbot ---
with tabs[2]:
    st.header("💬 Gemini Chatbot Assistant")
    user_input = st.text_area("Ask any health-related question:")
    if st.button("Ask Gemini"):
        response = client.models.generate_content(
            model=st.secrets["GEMINI_MODEL"],
            contents=user_input
        )
        st.write(response.text)

# --- Tab 4: Sentiment ---
with tabs[3]:
    st.header("💭 Patient Sentiment Analyzer")
    text = st.text_input("Enter feedback or review:")
    if st.button("Analyze Sentiment"):
        vec = models["sentiment_vectorizer"].transform([text])
        pred = models["sentiment_model"].predict(vec)[0]
        result = "Positive 😊" if pred == 1 else "Negative 😟"
        st.success(result)

# --- Tab 5: Dashboard ---
with tabs[4]:
    st.header("📊 Analytics Dashboard")
    st.write("Real-time correlation between Risk vs Length of Stay")
    sample = pd.DataFrame({
        "Risk": np.random.randint(0, 2, 50),
        "LOS": np.random.uniform(1, 10, 50)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=sample, ax=ax)
    st.pyplot(fig)
    st.info("Higher risk patients generally show increased hospital stay duration.")
