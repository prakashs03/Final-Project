import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import google.generativeai as genai
from textblob import TextBlob

# -----------------------------
# 1. Load secrets and Gemini setup
# -----------------------------
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="wide")
st.title("🏥 HealthAI - Smart Healthcare Assistant")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def download_from_drive(file_id, save_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

os.makedirs("models", exist_ok=True)

# -----------------------------
# 2. Load models from Drive
# -----------------------------
try:
    download_from_drive(st.secrets["GOOGLE_DRIVE_FILEID_RISK"], "models/risk_classifier_v1.joblib")
    download_from_drive(st.secrets["GOOGLE_DRIVE_FILEID_LOS"], "models/los_regressor_v1.joblib")
    download_from_drive(st.secrets["GOOGLE_DRIVE_FILEID_SENTIMENT_MODEL"], "models/sentiment_model.joblib")
    download_from_drive(st.secrets["GOOGLE_DRIVE_FILEID_SENTIMENT_VECTORIZER"], "models/sentiment_vectorizer.joblib")

    risk_model = joblib.load("models/risk_classifier_v1.joblib")
    los_model = joblib.load("models/los_regressor_v1.joblib")
    sent_model = joblib.load("models/sentiment_model.joblib")
    sent_vectorizer = joblib.load("models/sentiment_vectorizer.joblib")

    st.success("✅ All models loaded successfully from Google Drive!")
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")

# -----------------------------
# 3. Translator (Gemini)
# -----------------------------
def translate_text(text, target_lang):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(f"Translate to {target_lang}: {text}")
        return response.text
    except Exception as e:
        return f"Translation error: {e}"

# -----------------------------
# 4. Chatbot (Gemini)
# -----------------------------
def health_chatbot(query, explain=False):
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    if not explain:
        prompt = f"Give only 2–3 key points to answer briefly: {query}"
    else:
        prompt = f"Explain in detail about: {query}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Chatbot error: {e}"

# -----------------------------
# 5. Sentiment Analysis
# -----------------------------
def analyze_sentiment(text):
    X = sent_vectorizer.transform([text])
    pred = sent_model.predict(X)[0]
    sentiment = "😊 Positive" if pred == 1 else "😠 Negative"
    polarity = TextBlob(text).sentiment.polarity
    return sentiment, round(polarity, 2)

# -----------------------------
# 6. Streamlit Tabs
# -----------------------------
tabs = st.tabs(["🏥 Risk Prediction", "📊 LOS Prediction", "💬 Chatbot", "🌐 Translator", "❤️ Sentiment Analysis"])

# --- Risk Prediction ---
with tabs[0]:
    st.header("Disease Risk Classification")
    age = st.number_input("Age", 0, 100, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)

    if st.button("Predict Risk"):
        try:
            pred = risk_model.predict([[age, bp, glucose, bmi]])[0]
            st.success(f"🩸 Risk Category: {'High Risk' if pred==1 else 'Low Risk'}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- LOS Regression ---
with tabs[1]:
    st.header("Length of Stay (Regression)")
    age = st.number_input("Age", 0, 100, 50, key="age2")
    disease_score = st.slider("Disease Severity", 0, 10, 5)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key="bmi2")

    if st.button("Predict LOS"):
        try:
            los = los_model.predict([[age, disease_score, bmi]])[0]
            st.info(f"🏥 Predicted Length of Stay: {los:.2f} days")
        except Exception as e:
            st.error(f"Regression error: {e}")

# --- Chatbot ---
with tabs[2]:
    st.header("AI Health Chatbot (Gemini)")
    user_q = st.text_input("Ask a medical question")
    if st.button("Get Short Answer"):
        st.write(health_chatbot(user_q, explain=False))
    if st.button("Explain in Detail"):
        st.write(health_chatbot(user_q, explain=True))

# --- Translator ---
with tabs[3]:
    st.header("Translate Medical Information 🌐")
    text = st.text_area("Enter text to translate")
    lang = st.selectbox("Choose target language", ["Tamil", "Hindi", "Malayalam", "English"])
    if st.button("Translate"):
        st.write(translate_text(text, lang))

# --- Sentiment Analysis ---
with tabs[4]:
    st.header("Patient Feedback Sentiment 💬")
    feedback = st.text_area("Enter feedback")
    if st.button("Analyze Sentiment"):
        label, score = analyze_sentiment(feedback)
        st.write(f"Sentiment: {label} | Polarity Score: {score}")
