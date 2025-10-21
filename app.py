# ========================== HEALTHAI — FINAL PROJECT ==========================
# Author: Jayaprakash Srinivasan
# Description: End-to-End Smart Healthcare System with ML, DL & Gemini AI
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import google.generativeai as genai
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from nltk.translate.bleu_score import sentence_bleu
import math

# ---------------------- GEMINI CONFIG ----------------------
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
GENAI_MODEL = st.secrets["GENAI_MODEL"]
genai.configure(api_key=GENAI_API_KEY)
gemini = genai.GenerativeModel(GENAI_MODEL)

# ---------------------- PAGE CONFIG ------------------------
st.set_page_config(page_title="💊 HealthAI Dashboard", layout="wide")
st.title("💊 HealthAI — AI-Powered Clinical Decision Support System")
st.caption("Designed & Developed by **Jayaprakash Srinivasan**")

# ---------------------- LOAD MODELS ------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["risk"] = joblib.load("models/risk_classifier_v1.joblib")
        models["los"] = joblib.load("models/los_regressor_v1.joblib")
        models["cluster"] = joblib.load("models/patient_cluster_model.joblib")
        models["cnn"] = tf.keras.models.load_model("models/cnn_model_v1.h5", compile=False)
        models["lstm"] = tf.keras.models.load_model("models/lstm_model_v1.h5", compile=False)
        models["sentiment_model"] = joblib.load("models/sentiment_model.joblib")
        models["sentiment_vectorizer"] = joblib.load("models/sentiment_vectorizer.joblib")
        models["risk_scaler"] = joblib.load("models/risk_scaler.joblib")
        models["los_scaler"] = joblib.load("models/los_scaler.joblib")
        models["lstm_scaler"] = joblib.load("models/lstm_scaler.joblib")
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
    return models

models = load_models()

# ---------------------- HELPER FUNCTIONS -------------------
def preprocess_image(img, model):
    img = img.convert("RGB").resize((model.input_shape[1], model.input_shape[2]))
    arr = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def cnn_predict(img):
    arr = preprocess_image(img, models["cnn"])
    preds = models["cnn"].predict(arr)
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
        label = "Pneumonia" if prob >= 0.5 else "Normal"
    else:
        idx = int(np.argmax(preds))
        classes = ["Normal", "Pneumonia"]
        label = classes[idx]
        prob = float(np.max(preds))
    return label, round(prob * 100, 2)

def predict_los(features):
    try:
        scaled = models["los_scaler"].transform([features])
        y_scaled = models["los"].predict(scaled).reshape(-1, 1)
        inv = models["los_scaler"].inverse_transform(
            np.hstack([y_scaled] * models["los_scaler"].scale_.shape[0])
        )[0][0]
        return max(1, round(float(inv), 1))
    except Exception:
        return max(1, round(float(models["los"].predict([features])[0]), 1))

def lstm_forecast(series):
    try:
        series = np.asarray(series).astype(np.float32)
        _, timesteps, features = models["lstm"].input_shape
        if features is None:
            features = 1
        if series.size < timesteps:
            pad = np.full(timesteps - series.size, series[-1])
            series = np.concatenate([pad, series])
        series = series[-timesteps:]
        X = series.reshape(1, timesteps, features)
        pred_scaled = models["lstm"].predict(X)
        val = models["lstm_scaler"].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        return round(float(val), 2)
    except Exception:
        return round(float(np.mean(series)), 2)

def gemini_reply(prompt):
    try:
        res = gemini.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"(Gemini Error) {e}"

def sentiment_predict(text):
    X = models["sentiment_vectorizer"].transform([text])
    pred = models["sentiment_model"].predict(X)[0]
    sentiment_label = "Positive 😀" if pred == 1 else "Negative 😞"
    gemini_check = gemini_reply(f"Perform sentiment analysis: {text}. Answer Positive or Negative only.")
    if "Positive" in gemini_check:
        sentiment_label = "Positive 😀"
    elif "Negative" in gemini_check:
        sentiment_label = "Negative 😞"
    return sentiment_label

# ---------------------- TABS -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 Risk & Stay Prediction", 
    "🧠 LSTM Forecast", 
    "🩻 CNN Diagnostics", 
    "💬 Chatbot", 
    "🌐 Translator", 
    "❤️ Sentiment Analysis",
    "📊 Evaluation Metrics"
])

# ---------------------- TAB 1: RISK + STAY -----------------
with tab1:
    st.subheader("🏥 Disease Risk Classification & Hospital Stay Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 100)
        bp = st.number_input("Blood Pressure", 50, 200)
    with c2:
        chol = st.number_input("Cholesterol", 50, 400)
        sugar = st.number_input("Blood Sugar", 50, 300)
    with c3:
        bmi = st.number_input("BMI", 10.0, 50.0)
        heart_rate = st.number_input("Heart Rate", 30, 180)

    if st.button("🔍 Predict Risk & Stay"):
        features = [age, bp, chol, sugar, bmi, heart_rate]
        risk_scaled = models["risk_scaler"].transform([features])
        risk_pred = models["risk"].predict(risk_scaled)[0]
        los_pred = predict_los(features)

        st.success(f"🩺 Disease Risk: {'High' if risk_pred == 1 else 'Low'}")
        st.info(f"🏨 Expected Hospital Stay: {los_pred} days")

# ---------------------- TAB 2: LSTM FORECAST ----------------
with tab2:
    st.subheader("🧠 Health Metric Forecast using LSTM")
    seq = st.text_area("Enter comma-separated patient vitals (e.g., 98,99,100,101)")
    if st.button("📈 Forecast"):
        if seq:
            arr = list(map(float, seq.split(",")))
            forecast = lstm_forecast(arr)
            st.metric("Predicted Metric", f"{forecast}")
        else:
            st.warning("Enter numeric sequence data!")

# ---------------------- TAB 3: CNN DIAGNOSTICS --------------
with tab3:
    st.subheader("🩻 Pneumonia Detection via CNN")
    file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", width=250)
        if st.button("🧠 Analyze Image"):
            label, prob = cnn_predict(img)
            st.success(f"Prediction: **{label}** ({prob}%)")

# ---------------------- TAB 4: CHATBOT ----------------------
with tab4:
    st.subheader("💬 Gemini Healthcare Chatbot")
    query = st.text_input("Ask any health-related question:")
    if query:
        short = gemini_reply(f"Answer in 5 words only: {query}")
        st.info(f"💡 Quick Answer: {short}")
        if st.button("Explain More"):
            detailed = gemini_reply(f"Explain shortly: {query}")
            st.success(detailed)

# ---------------------- TAB 5: TRANSLATOR -------------------
with tab5:
    st.subheader("🌐 Multilingual Medical Translator")
    text = st.text_input("Enter medical sentence:")
    lang = st.text_input("Translate to (e.g., Tamil, Hindi, French)")
    if st.button("🌍 Translate"):
        result = gemini_reply(f"Translate this medical text into {lang}: {text}")
        st.success(result)

# ---------------------- TAB 6: SENTIMENT --------------------
with tab6:
    st.subheader("❤️ Patient Feedback Sentiment (Gemini + ML)")
    feedback = st.text_area("Enter feedback:")
    if st.button("🧭 Analyze Sentiment"):
        if feedback.strip():
            result = sentiment_predict(feedback)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter feedback text.")

# ---------------------- TAB 7: EVALUATION METRICS -----------
with tab7:
    st.subheader("📊 Evaluation Metrics Dashboard")

    # Dummy demonstration values
    y_true_class = [0, 1, 1, 0, 1]
    y_pred_class = [0, 1, 0, 0, 1]
    y_true_reg = [2.3, 3.1, 4.5, 2.7]
    y_pred_reg = [2.5, 3.0, 4.3, 2.8]

    acc = accuracy_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse = math.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    bleu = sentence_bleu([["the", "patient", "is", "stable"]], ["patient", "is", "stable"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("F1 Score", f"{f1:.2f}")
    c3.metric("MAE", f"{mae:.2f}")
    c4.metric("RMSE", f"{rmse:.2f}")
    c5.metric("BLEU", f"{bleu:.2f}")

    st.caption("✅ Metrics are sample demo values; models can be validated with test datasets.")
