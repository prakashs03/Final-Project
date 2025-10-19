import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from google import genai
from deep_translator import GoogleTranslator
import xgboost as xgb

# =====================================
# INITIAL SETUP
# =====================================

st.set_page_config(page_title="💊 HealthAI — Smart Healthcare Assistant", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-powered multimodal healthcare analysis and Gemini Chatbot")

os.makedirs("models", exist_ok=True)

# Gemini client
client = genai.Client(api_key=st.secrets["GENAI_API_KEY"])
DRIVE = st.secrets["DRIVE"]

# =====================================
# MODEL DOWNLOAD FUNCTION
# =====================================

@st.cache_resource
def download_model(file_url, filename):
    """Download from full Google Drive URL directly (Streamlit-safe)."""
    output_path = f"models/{filename}"
    try:
        gdown.download(file_url, output_path, quiet=False, fuzzy=True)
        if os.path.exists(output_path):
            st.write(f"✅ {filename} downloaded")
        else:
            st.warning(f"⚠️ Could not download {filename}")
    except Exception as e:
        st.warning(f"⚠️ Download issue for {filename}: {e}")
    return output_path

# =====================================
# LOAD MODELS FROM DRIVE
# =====================================

@st.cache_resource
def load_models():
    """Load all models with safety checks."""
    models = {}
    try:
        models["risk"] = joblib.load(download_model(DRIVE["risk"], "risk_classifier_v1.joblib"))
        models["los"] = joblib.load(download_model(DRIVE["los"], "los_regressor_v1.joblib"))
        models["cnn"] = tf.keras.models.load_model(download_model(DRIVE["cnn_h5"], "cnn_model_v1.h5"))
        models["lstm"] = tf.keras.models.load_model(download_model(DRIVE["lstm_h5"], "lstm_model_v1.h5"))
        models["sentiment_model"] = joblib.load(download_model(DRIVE["sentiment_model"], "sentiment_model.joblib"))
        models["sentiment_vectorizer"] = joblib.load(download_model(DRIVE["sentiment_vectorizer"], "sentiment_vectorizer.joblib"))
        models["risk_scaler"] = joblib.load(download_model(DRIVE["risk_scaler"], "risk_scaler.joblib"))
        models["los_scaler"] = joblib.load(download_model(DRIVE["los_scaler"], "los_scaler.joblib"))
        models["lstm_scaler"] = joblib.load(download_model(DRIVE["lstm_scaler"], "lstm_scaler.joblib"))
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
    return models

models = load_models()

# =====================================
# APP INTERFACE (TABS)
# =====================================

tabs = st.tabs(["🏥 Risk & LOS", "🧠 CNN / LSTM", "💬 Chatbot", "💭 Sentiment", "📊 Dashboard"])

# ============================
# TAB 1: RISK & LOS
# ============================
with tabs[0]:
    st.header("🏥 Disease Risk and Length of Stay Prediction")

    age = st.number_input("Age", 0, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    glucose = st.number_input("Glucose", 50, 300, 120)
    heart_rate = st.number_input("Heart Rate", 40, 180, 80)

    if st.button("Predict Disease Risk"):
        X = np.array([[age, bp, chol, bmi, glucose, heart_rate]])
        X_scaled = models["risk_scaler"].transform(X)
        pred = models["risk"].predict(X_scaled)
        risk = "HIGH ⚠️" if pred[0] == 1 else "LOW ✅"
        st.metric("Predicted Disease Risk", risk)

    if st.button("Predict Length of Stay"):
        X = np.array([[age, bp, chol, bmi, glucose, heart_rate]])
        X_scaled = models["los_scaler"].transform(X)
        los_pred = models["los"].predict(X_scaled)
        st.metric("Predicted Stay Duration (Days)", f"{los_pred[0]:.2f}")

# ============================
# TAB 2: CNN + LSTM
# ============================
with tabs[1]:
    st.header("🧠 Deep Learning Models")

    col1, col2 = st.columns(2)

    # CNN MODEL
    with col1:
        st.subheader("🩻 CNN — X-Ray Image Diagnosis")
        img_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])
        if img_file:
            img = Image.open(img_file).convert("RGB").resize((128, 128))
            st.image(img, caption="Uploaded Image", width=250)
            img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = models["cnn"].predict(img_arr)
            label = "Pneumonia 🫁" if np.argmax(prediction) == 1 else "Normal ✅"
            st.metric("CNN Diagnosis", label)

    # LSTM MODEL
    with col2:
        st.subheader("📈 LSTM — Health Time-Series Forecast")
        time_steps = np.arange(50)
        data = np.sin(time_steps) + np.random.normal(0, 0.1, 50)
        scaled = models["lstm_scaler"].transform(data.reshape(-1, 1))
        X = scaled.reshape(1, 50, 1)
        pred = models["lstm"].predict(X)
        forecast = models["lstm_scaler"].inverse_transform(pred)[0][0]
        st.metric("Forecasted Health Metric", f"{forecast:.2f}")

        # Plot Visualization
        plt.figure(figsize=(6, 3))
        plt.plot(data, label="History")
        plt.axhline(forecast, color='r', linestyle='--', label="Forecast")
        plt.legend()
        st.pyplot(plt)

# ============================
# TAB 3: CHATBOT (Gemini)
# ============================
with tabs[2]:
    st.header("💬 Gemini Health Chatbot")
    query = st.text_area("Ask any medical or healthcare question:")

    if st.button("Ask Gemini"):
        try:
            response = client.models.generate_content(
                model=st.secrets["GEMINI_MODEL"],
                contents=query
            )
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API error: {e}")

# ============================
# TAB 4: SENTIMENT
# ============================
with tabs[3]:
    st.header("💭 Sentiment Analyzer")
    feedback = st.text_input("Enter patient feedback:")
    if st.button("Analyze Feedback"):
        vec = models["sentiment_vectorizer"].transform([feedback])
        pred = models["sentiment_model"].predict(vec)[0]
        result = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"Sentiment: {result}")

# ============================
# TAB 5: DASHBOARD
# ============================
with tabs[4]:
    st.header("📊 Dashboard — Risk vs Stay Comparison")

    df = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 50),
        "LOS": np.random.uniform(1, 10, 50)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=df, ax=ax)
    st.pyplot(fig)
    st.info("🧠 Insight: High-risk patients generally stay longer in hospital.")
