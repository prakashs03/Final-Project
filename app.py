import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, cv2, gdown
import tensorflow as tf
from textblob import TextBlob
from deep_translator import GoogleTranslator
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import google.generativeai as genai

# ===================== CONFIG =====================
st.set_page_config(page_title="HealthAI Platform", layout="wide")
st.title("🏥 HealthAI: End-to-End Healthcare AI Platform")
st.caption("Classification | Regression | CNN | LSTM | Translator | Chatbot | Sentiment (Gemini)")

# ===================== GEMINI SETUP =====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True
except Exception:
    st.warning("⚠️ GEMINI_API_KEY missing — Translator, Chatbot, Sentiment disabled.")
    gemini_available = False

# ===================== GOOGLE DRIVE MODELS =====================
os.makedirs("models", exist_ok=True)

drive_links = st.secrets["DRIVE"]
download_map = {
    "models/rf_classifier.joblib": drive_links.get("risk"),
    "models/rf_regressor.joblib": drive_links.get("los"),
    "models/cnn_best.h5": drive_links.get("cnn_h5"),
    "models/lstm_best.h5": drive_links.get("lstm_h5"),
    "models/sentiment_model.h5": drive_links.get("sentiment_model")
}

for path, url in download_map.items():
    if url and not os.path.exists(path):
        try:
            st.info(f"📥 Downloading {os.path.basename(path)} from Drive...")
            gdown.download(url, path, quiet=False)
            st.success(f"✅ {os.path.basename(path)} downloaded successfully.")
        except Exception as e:
            st.warning(f"⚠️ Could not download {os.path.basename(path)} — {e}")

# ===================== MENU =====================
menu = st.sidebar.radio(
    "Select Module",
    [
        "Classification",
        "Regression",
        "Clustering",
        "Association Rules",
        "CNN Imaging",
        "LSTM Forecasting",
        "Translator",
        "Chatbot",
        "Sentiment"
    ]
)

# ===================== UTILS =====================
def preprocess(df):
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"M": 1, "F": 0}).fillna(0)
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(df.median(numeric_only=True))
    return df

# ===================== CLASSIFICATION =====================
if menu == "Classification":
    st.header("🧬 Disease Risk Classification")
    file = st.file_uploader("Upload labeled CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        try:
            model = joblib.load("models/rf_classifier.joblib")
            X = preprocess(df)
            preds = model.predict(X)
            st.success("✅ Predictions Generated!")
            st.bar_chart(pd.Series(preds).value_counts())
        except Exception as e:
            st.error(f"Model load error: {e}")

# ===================== REGRESSION =====================
elif menu == "Regression":
    st.header("📈 Length of Stay Prediction")
    file = st.file_uploader("Upload CSV for LOS prediction", type=["csv"])
    if file:
        df = pd.read_csv(file)
        try:
            model = joblib.load("models/rf_regressor.joblib")
            X = preprocess(df)
            preds = model.predict(X)
            preds = np.clip(preds, 1, 30)
            st.success("✅ LOS Predictions Generated!")
            st.write("Predicted LOS (first 10):", preds[:10])
            fig = px.histogram(preds, nbins=15, title="Distribution of Predicted Stay (Days)")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ===================== CNN IMAGING =====================
elif menu == "CNN Imaging":
    st.header("🩻 Chest X-Ray Classifier (Normal vs Pneumonia)")
    file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])
    if file:
        img = tf.keras.utils.load_img(file, target_size=(224, 224))
        img_arr = tf.keras.utils.img_to_array(img)
        img_arr = np.expand_dims(img_arr / 255.0, axis=0)
        model = tf.keras.models.load_model("models/cnn_best.h5")
        preds = model.predict(img_arr)
        label_index = np.argmax(preds)
        confidence = float(np.max(preds))
        label = "PNEUMONIA" if label_index == 1 else "NORMAL"
        color = "🔴" if label == "PNEUMONIA" else "🟢"
        st.image(file, caption=f"{color} Prediction: {label} ({confidence*100:.2f}% confidence)", width=350)

# ===================== LSTM FORECASTING =====================
elif menu == "LSTM Forecasting":
    st.header("📊 Patient Vital Trend Forecasting (LSTM)")
    st.write("This module forecasts patient metrics using an LSTM network.")
    data = np.linspace(0, 4*np.pi, 100)
    vitals = np.sin(data) + np.random.normal(0, 0.1, 100)
    X, y = [], []
    for i in range(len(vitals) - 10):
        X.append(vitals[i:i+10])
        y.append(vitals[i+10])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(X, y, epochs=5, verbose=0)
    preds = model.predict(X)
    st.success("✅ LSTM Forecast Completed")
    fig = px.line(x=range(len(preds)), y=preds[:, 0], title="Forecasted Patient Vital Trend")
    st.plotly_chart(fig)

# ===================== CHATBOT (Gemini) =====================
elif menu == "Chatbot":
    st.header("💬 Healthcare Chatbot (Gemini)")
    user_input = st.text_input("Ask your question:")
    if st.button("Send"):
        if gemini_available:
            try:
                short_prompt = f"Give a 2-word medical summary: {user_input}"
                detailed_prompt = f"Explain in detail medically but safely: {user_input}"
                short_resp = genai.responses.create(model="models/gemini-2.5-flash", input=short_prompt)
                short_out = "".join([c.text for o in short_resp.output for c in o.content if c.text])
                st.info(f"🩺 Short Answer: {short_out.strip()}")
                if "explain" in user_input.lower() or "detail" in user_input.lower():
                    det_resp = genai.responses.create(model="models/gemini-2.5-pro", input=detailed_prompt)
                    detail_out = "".join([c.text for o in det_resp.output for c in o.content if c.text])
                    st.success(f"📖 Detailed: {detail_out.strip()}")
            except Exception as e:
                st.error(f"Chatbot error: {e}")
        else:
            st.warning("Gemini API not found in secrets.")

# ===================== TRANSLATOR =====================
elif menu == "Translator":
    st.header("🌍 Medical Translator (Gemini)")
    txt = st.text_area("Enter text to translate:")
    lang = st.selectbox("Target language:", ["ta", "hi", "fr", "es"])
    if st.button("Translate"):
        if gemini_available:
            try:
                prompt = f"Translate the following text to {lang}: {txt}"
                response = genai.responses.create(model="models/gemini-2.5-flash", input=prompt)
                output = "".join([c.text for o in response.output for c in o.content if c.text])
                st.success(output.strip())
            except Exception as e:
                st.error(f"Translation failed: {e}")

# ===================== SENTIMENT =====================
elif menu == "Sentiment":
    st.header("💭 Sentiment Analysis (Gemini)")
    text = st.text_area("Enter patient feedback:")
    if st.button("Analyze"):
        if gemini_available:
            try:
                prompt = f"Classify as Positive, Neutral, or Negative with 1-line reason: {text}"
                response = genai.responses.create(model="models/gemini-2.5-flash", input=prompt)
                output = "".join([c.text for o in response.output for c in o.content if c.text])
                st.success(output.strip())
            except Exception as e:
                st.error(f"Sentiment analysis error: {e}")
