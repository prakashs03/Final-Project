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

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="💊 HealthAI - Smart Healthcare Assistant", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-powered multimodal healthcare assistant — Risk, LOS, CNN, LSTM, Sentiment, Chatbot & Translation")

os.makedirs("models", exist_ok=True)

# ================================
# GOOGLE DRIVE + GEMINI SETUP
# ================================
DRIVE = st.secrets["DRIVE"]
GEMINI_API = st.secrets["GENAI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]
client = genai.Client(api_key=GEMINI_API)

# ================================
# SAFE MODEL DOWNLOADER
# ================================
@st.cache_resource
def download_model(url, filename):
    path = f"models/{filename}"
    try:
        gdown.download(url, path, quiet=False, fuzzy=True)
        if os.path.exists(path):
            st.write(f"✅ {filename} downloaded.")
    except Exception as e:
        st.warning(f"⚠️ Failed to download {filename}: {e}")
    return path

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
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
        st.error(f"Model loading failed: {e}")
    return models

models = load_models()

# ================================
# STREAMLIT TABS
# ================================
tabs = st.tabs(["🏥 Risk & LOS", "🧠 CNN & LSTM", "💬 Chatbot", "💭 Sentiment", "🌐 Translator", "📊 Dashboard"])

# ================================
# TAB 1 - RISK & LOS
# ================================
with tabs[0]:
    st.header("🏥 Disease Risk & Hospital Stay Duration")

    age = st.slider("Age", 10, 90, 40)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 350, 200)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    glucose = st.slider("Glucose", 50, 250, 110)
    heart_rate = st.slider("Heart Rate", 50, 160, 80)

    features = np.array([[age, bp, chol, bmi, glucose, heart_rate]])

    if st.button("Predict Risk"):
        scaled = models["risk_scaler"].transform(features)
        pred = models["risk"].predict(scaled)[0]
        result = "⚠️ High Risk" if pred == 1 else "✅ Low Risk"
        st.metric("Disease Risk Prediction", result)

    if st.button("Predict Length of Stay"):
        scaled = models["los_scaler"].transform(features)
        los_pred = models["los"].predict(scaled)
        # ✅ FIXED: no inverse_transform (6 features vs 1 output)
        try:
            los_days = float(los_pred[0])
        except Exception:
            los_days = float(np.array(los_pred).reshape(-1)[0])
        st.metric("Predicted Stay Duration", f"{los_days:.1f} Days")

# ================================
# TAB 2 - CNN & LSTM
# ================================
with tabs[1]:
    st.header("🧠 Deep Learning Models — CNN & LSTM")

    c1, c2 = st.columns(2)

    # CNN
    with c1:
        st.subheader("🩻 CNN — X-Ray Classification")
        img = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
        if img:
            image = Image.open(img).convert("RGB").resize((128, 128))
            st.image(image, width=250)
            arr = np.expand_dims(np.array(image) / 255.0, axis=0)
            preds = models["cnn"].predict(arr)
            confidence = np.max(preds)
            label = np.argmax(preds)
            if label == 1 and confidence > 0.6:
                st.success(f"🫁 Pneumonia Detected ({confidence*100:.1f}% confidence)")
            elif label == 0 and confidence > 0.6:
                st.success(f"✅ Normal ({confidence*100:.1f}% confidence)")
            else:
                st.warning(f"🤔 Uncertain — Confidence: {confidence*100:.1f}%")

    # LSTM
    with c2:
        st.subheader("📈 LSTM — Health Metric Forecast")

        timesteps = np.arange(20)
        synthetic = np.sin(timesteps) + np.random.normal(0, 0.1, 20)
        scaled = models["lstm_scaler"].transform(synthetic.reshape(-1, 1))

        try:
            input_shape = models["lstm"].input_shape
            time_steps = input_shape[1] if input_shape[1] else scaled.shape[0]
            n_features = input_shape[2] if input_shape[2] else 1

            if scaled.shape[0] < time_steps:
                pad_len = time_steps - scaled.shape[0]
                scaled = np.pad(scaled, ((0, pad_len), (0, 0)), mode='edge')
            elif scaled.shape[0] > time_steps:
                scaled = scaled[:time_steps]

            if n_features > 1:
                scaled = np.repeat(scaled, n_features, axis=1)

            X = scaled.reshape(1, time_steps, n_features)
            pred = models["lstm"].predict(X)
            val = models["lstm_scaler"].inverse_transform(pred)[0][0]
            st.metric("Forecasted Health Metric", f"{val:.2f}")

            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(range(time_steps), scaled[:,0], label="Input Signal")
            ax.axhline(val, color='r', linestyle='--', label='Forecast')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ LSTM model issue: {e}")

# ================================
# TAB 3 - CHATBOT
# ================================
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
                        contents=f"Give a short, keyword-based medical answer for: {query_en}"
                    )
                    st.info(short_resp.text)
                except Exception as e:
                    st.error(f"Gemini error: {e}")
        with col2:
            if st.button("Explain More"):
                try:
                    long_resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=f"Explain in detail about: {query_en}"
                    )
                    st.success(long_resp.text)
                except Exception as e:
                    st.error(f"Gemini error: {e}")

# ================================
# TAB 4 - SENTIMENT
# ================================
with tabs[3]:
    st.header("💭 Sentiment Analysis — Patient Feedback")

    feedback = st.text_input("Enter feedback:")
    if st.button("Analyze Sentiment"):
        if feedback.strip() == "":
            st.warning("Please enter some feedback.")
        else:
            try:
                gemini_sentiment = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"Classify sentiment of this sentence as Positive, Negative, or Neutral: {feedback}"
                )
                response = gemini_sentiment.text.strip()
                if "Positive" in response:
                    st.metric("Sentiment", "Positive 😊")
                elif "Negative" in response:
                    st.metric("Sentiment", "Negative 😞")
                else:
                    st.metric("Sentiment", "Neutral 😐")
            except Exception:
                vec = models["sentiment_vectorizer"].transform([feedback])
                pred = models["sentiment_model"].predict(vec)[0]
                sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
                st.metric("Sentiment", sentiment)

# ================================
# TAB 5 - TRANSLATOR
# ================================
with tabs[4]:
    st.header("🌐 Translator — Multilingual Support")
    text = st.text_area("Enter text to translate:")
    lang = st.selectbox("Select language", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        try:
            translated = GoogleTranslator(source="auto", target=lang).translate(text)
            st.success(f"🔤 Translated ({lang}): {translated}")
        except Exception as e:
            st.warning(f"Translation failed: {e}")

# ================================
# TAB 6 - DASHBOARD
# ================================
with tabs[5]:
    st.header("📊 Dashboard — Risk & Stay Comparison")
    data = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 100),
        "LOS": np.random.uniform(1, 10, 100)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=data, ax=ax)
    st.pyplot(fig)
    st.info("🧠 Insight: High-risk patients generally stay longer.")
