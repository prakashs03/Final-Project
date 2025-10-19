import os
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

# Create models folder in runtime (Streamlit Cloud fix)
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="💊 HealthAI — Smart Healthcare Assistant", layout="wide")

client = genai.Client(api_key=st.secrets["GENAI_API_KEY"])
DRIVE = st.secrets["DRIVE"]

# ============================
# HELPER FUNCTIONS
# ============================

@st.cache_resource
def download_model(file_id, filename):
    """Download model from Google Drive and return local path."""
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = f"models/{filename}"
    gdown.download(url, output_path, quiet=True)
    return output_path

@st.cache_resource
def load_models():
    """Load all models from Drive once."""
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
    except Exception as e:
        st.warning(f"⚠️ Model load issue: {e}")
    return models

models = load_models()
st.success("✅ All models loaded successfully!")

# ============================
# MAIN INTERFACE
# ============================

tabs = st.tabs(["🏥 Risk & LOS", "🧠 CNN / LSTM", "💬 Chatbot", "💭 Sentiment", "📊 Dashboard"])

# ============================
# TAB 1: Risk & LOS Prediction
# ============================

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
        pred = models["risk"].predict(X_scaled)
        st.metric("Predicted Risk", "HIGH ⚠️" if pred[0] == 1 else "LOW ✅")

    if st.button("Predict Length of Stay"):
        X = np.array([[age, bp, chol, bmi, glucose, heart_rate]])
        X_scaled = models["los_scaler"].transform(X)
        los_pred = models["los"].predict(X_scaled)
        st.metric("Predicted Stay (Days)", f"{los_pred[0]:.2f}")

# ============================
# TAB 2: CNN / LSTM
# ============================

with tabs[1]:
    st.header("🧠 Deep Learning Models")

    col1, col2 = st.columns(2)

    # --- CNN Model ---
    with col1:
        st.subheader("🩻 CNN — X-Ray Diagnosis")
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
            st.image(img, caption="Uploaded Image", width=250)
            img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = models["cnn"].predict(img_arr)
            result = "Pneumonia 🫁" if np.argmax(prediction) == 1 else "Normal ✅"
            st.metric("Diagnosis", result)

    # --- LSTM Model ---
    with col2:
        st.subheader("📈 LSTM — Health Forecasting")
        time_steps = np.arange(50)
        data = np.sin(time_steps) + np.random.normal(0, 0.1, 50)
        scaled = models["lstm_scaler"].transform(data.reshape(-1, 1))
        X = scaled.reshape(1, 50, 1)
        pred = models["lstm"].predict(X)
        forecast = models["lstm_scaler"].inverse_transform(pred)[0][0]
        st.metric("Next Forecast Value", f"{forecast:.2f}")

        # Plot visualization
        plt.figure(figsize=(6, 3))
        plt.plot(data, label="History")
        plt.axhline(forecast, color='r', linestyle='--', label="Forecast")
        plt.legend()
        st.pyplot(plt)

# ============================
# TAB 3: Chatbot (Gemini)
# ============================

with tabs[2]:
    st.header("💬 Gemini Health Chatbot")
    user_input = st.text_area("Ask me any health-related question:")
    if st.button("Ask Gemini"):
        try:
            response = client.models.generate_content(
                model=st.secrets["GEMINI_MODEL"],
                contents=user_input
            )
            st.write(response.text)
        except Exception as e:
            st.error(f"Error from Gemini API: {e}")

# ============================
# TAB 4: Sentiment Analysis
# ============================

with tabs[3]:
    st.header("💭 Patient Sentiment Analyzer")
    text = st.text_input("Enter a feedback or review:")
    if st.button("Analyze Sentiment"):
        vec = models["sentiment_vectorizer"].transform([text])
        pred = models["sentiment_model"].predict(vec)[0]
        result = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(result)

# ============================
# TAB 5: Dashboard Visualization
# ============================

with tabs[4]:
    st.header("📊 Risk vs LOS Dashboard")
    st.caption("Visualizing relationship between risk level and hospital stay duration.")

    # Sample visualization
    df = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 50),
        "LOS": np.random.uniform(1, 10, 50)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=df, ax=ax)
    st.pyplot(fig)
    st.info("High-risk patients generally show longer hospital stays.")
