import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import google.generativeai as genai
from PIL import Image
import io
import os

# -----------------------------
# ✅ TensorFlow Compatibility Patch
# -----------------------------
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import InputLayer

@register_keras_serializable()
class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        super().__init__(*args, **kwargs)

# -----------------------------
# ✅ Gemini API Setup
# -----------------------------
genai.configure(api_key=st.secrets["GENAI_API_KEY"])
model_gemini = genai.GenerativeModel(st.secrets["GENAI_MODEL"])

# -----------------------------
# ✅ Load Models from Local / GitHub
# -----------------------------
@st.cache_resource
def load_models_local():
    models = {}
    model_files = {
        "risk": "models/risk_classifier_v1.joblib",
        "los": "models/los_regressor_v1.joblib",
        "los_scaler": "models/los_scaler.joblib",
        "lstm": "models/lstm_model_v1.h5",
        "lstm_scaler": "models/lstm_scaler.joblib",
        "cnn": "models/cnn_model_v1.h5",
        "sentiment_model": "models/sentiment_model.joblib",
        "sentiment_vectorizer": "models/sentiment_vectorizer.joblib",
    }
    for key, path in model_files.items():
        try:
            if path.endswith(".joblib"):
                models[key] = joblib.load(path)
            elif path.endswith(".h5"):
                models[key] = tf.keras.models.load_model(path, compile=False)
            print(f"✅ Loaded {key}")
        except Exception as e:
            print(f"⚠️ Warning: failed loading {key}: {e}")
    return models

models = load_models_local()

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI — Smart Health Prediction Suite")

tabs = st.tabs([
    "🏥 Risk & LOS",
    "📈 LSTM Forecast",
    "🩻 CNN (X-ray)",
    "🤖 Chatbot (Gemini)",
    "🌐 Translator (Gemini)",
    "💬 Sentiment"
])

# =======================================================
# 🧮 Tab 1: Risk Classification + LOS Prediction
# =======================================================
with tabs[0]:
    st.header("Disease Risk Classification + Length of Stay Prediction")

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    hr = st.number_input("Heart Rate", 30, 200, 80)
    chol = st.number_input("Cholesterol", 100, 400, 200)

    if st.button("🔍 Predict Risk & Stay Duration"):
        features = np.array([[age, bp, glucose, bmi, hr, chol]])
        risk_pred = los_pred = None

        if "risk" in models:
            risk_pred = models["risk"].predict(features)
            risk_label = "High Risk" if risk_pred[0] == 1 else "Low Risk"
            st.metric("Disease Risk", risk_label)
        else:
            st.warning("Risk model unavailable")

        if "los" in models and "los_scaler" in models:
            try:
                scaled = models["los_scaler"].transform(features)
                los_pred = models["los"].predict(scaled)
                los_days = float(models["los_scaler"].inverse_transform(
                    np.array(los_pred).reshape(-1, 1)
                )[0][0])
                st.metric("Predicted Stay Duration", f"{los_days:.1f} Days")
            except Exception as e:
                st.error(f"LOS Prediction Error: {e}")
        else:
            st.warning("LOS model unavailable")

# =======================================================
# 📊 Tab 2: LSTM Forecast
# =======================================================
with tabs[1]:
    st.header("📈 LSTM Patient Vitals Forecasting")
    if "lstm" in models and "lstm_scaler" in models:
        uploaded = st.file_uploader("Upload vitals CSV (time series)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write("Uploaded Data", df.head())
            scaled = models["lstm_scaler"].transform(df.values)
            seq = np.expand_dims(scaled, axis=0)
            pred = models["lstm"].predict(seq)
            pred_rescaled = models["lstm_scaler"].inverse_transform(pred.reshape(-1, df.shape[1]))
            st.line_chart(pred_rescaled)
    else:
        st.warning("LSTM model unavailable")

# =======================================================
# 🩻 Tab 3: CNN Image Classification
# =======================================================
with tabs[2]:
    st.header("🩻 X-ray Disease Detection")
    img_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
    if img_file and "cnn" in models:
        image = Image.open(img_file).convert("RGB").resize((128, 128))
        arr = np.expand_dims(np.array(image) / 255.0, axis=0)
        preds = models["cnn"].predict(arr)
        label = "Pneumonia" if preds[0][0] > 0.5 else "Normal"
        st.image(image, caption=f"Prediction: {label}", use_container_width=True)
    else:
        st.warning("CNN model unavailable")

# =======================================================
# 🤖 Tab 4: Chatbot
# =======================================================
with tabs[3]:
    st.header("🤖 Healthcare Chatbot (Gemini)")
    user_q = st.text_input("Ask your medical question:")
    explain = st.checkbox("Need detailed explanation?")
    if st.button("Ask"):
        if user_q.strip():
            try:
                short_ans = model_gemini.generate_content(user_q + " (reply briefly)")
                st.markdown(f"**Answer:** {short_ans.text}")
                if explain:
                    long_ans = model_gemini.generate_content(user_q + " (explain in detail)")
                    st.info(long_ans.text)
            except Exception as e:
                st.error(f"Chatbot error: {e}")

# =======================================================
# 🌐 Tab 5: Translator
# =======================================================
with tabs[4]:
    st.header("🌐 Medical Translator (Gemini)")
    text = st.text_area("Enter text to translate:")
    lang = st.text_input("Target language (e.g., Tamil, Hindi, Telugu):")
    if st.button("Translate"):
        if text.strip() and lang.strip():
            try:
                resp = model_gemini.generate_content(
                    f"Translate this medical text to {lang}: {text}"
                )
                st.success(resp.text)
            except Exception as e:
                st.error(f"Translation error: {e}")

# =======================================================
# 💬 Tab 6: Sentiment
# =======================================================
with tabs[5]:
    st.header("💬 Patient Feedback Sentiment Analysis")
    feedback = st.text_area("Enter feedback:")
    if st.button("Analyze Sentiment"):
        if "sentiment_model" in models and "sentiment_vectorizer" in models:
            vec = models["sentiment_vectorizer"].transform([feedback])
            pred = models["sentiment_model"].predict(vec)[0]
            result = "Positive" if pred == 1 else "Negative"
            st.success(f"Sentiment: {result}")
        else:
            try:
                resp = model_gemini.generate_content(
                    f"Classify the sentiment (Positive/Negative) of: {feedback}"
                )
                st.info(resp.text)
            except Exception as e:
                st.error(f"Sentiment analysis error: {e}")

st.caption("© 2025 HealthAI — Powered by Streamlit & Gemini")
