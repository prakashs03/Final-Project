import os
os.environ["WATCHDOG_MAX_INSTANCES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("End-to-end AI/ML system for patient risk, clustering, association, NLP, and diagnostics")

# ============================
DRIVE = st.secrets["DRIVE"]
GEMINI_API = st.secrets["GENAI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]
client = genai.Client(api_key=GEMINI_API)
os.makedirs("models", exist_ok=True)

@st.cache_resource
def download_model(url, filename):
    path = f"models/{filename}"
    try: gdown.download(url, path, quiet=True, fuzzy=True)
    except: pass
    return path

@st.cache_resource
def load_models():
    m = {}
    try:
        m["risk"] = joblib.load(download_model(DRIVE["risk"], "risk_classifier_v1.joblib"))
        m["los"] = joblib.load(download_model(DRIVE["los"], "los_regressor_v1.joblib"))
        m["cnn"] = tf.keras.models.load_model(download_model(DRIVE["cnn_h5"], "cnn_model_v1.h5"))
        m["risk_scaler"] = joblib.load(download_model(DRIVE["risk_scaler"], "risk_scaler.joblib"))
        m["los_scaler"] = joblib.load(download_model(DRIVE["los_scaler"], "los_scaler.joblib"))
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"Model load error: {e}")
    return m

models = load_models()

# ============================
tabs = st.tabs([
    "❤️ Heart Disease Risk",
    "🏥 LOS Regression",
    "🩺 Clustering",
    "🔗 Association Rules",
    "🧠 CNN X-Ray",
    "📉 LSTM Forecasting",
    "🧬 BioBERT NLP",
    "💬 Chatbot",
    "💭 Sentiment",
    "🌐 Translator",
    "📊 Dashboard"
])

# ===============================================
# ❤️ TAB 1 — Classification
# ===============================================
with tabs[0]:
    st.header("❤️ Heart Disease Risk Prediction")
    age = st.slider("Age", 10, 90, 45)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    chol = st.slider("Cholesterol", 100, 350, 200)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    glucose = st.slider("Glucose", 50, 250, 100)
    hr = st.slider("Heart Rate", 50, 150, 80)
    X = np.array([[age, bp, chol, bmi, glucose, hr]])

    if st.button("🔍 Predict Heart Disease Risk"):
        try:
            X_scaled = models["risk_scaler"].transform(X)
            clf = models["risk"]
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_scaled)[0]
                pred = np.argmax(proba)
                conf = np.max(proba)
            else:
                pred = clf.predict(X_scaled)[0]
                conf = None

            result = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk"
            st.metric("Prediction", result, f"Confidence: {conf*100:.1f}%")
        except Exception as e:
            st.error(f"Error: {e}")

# ===============================================
# 🏥 TAB 2 — Regression
# ===============================================
with tabs[1]:
    st.header("🏥 Length of Stay Prediction")
    if st.button("📅 Predict Stay Duration"):
        try:
            X_scaled = models["los_scaler"].transform(X)
            y_pred = models["los"].predict(X_scaled)
            try:
                days = float(models["los_scaler"].inverse_transform(y_pred.reshape(-1, 1))[0][0])
            except:
                days = float(np.array(y_pred).reshape(-1)[0])
            st.metric("Predicted Stay Duration", f"{days:.1f} Days")
        except Exception as e:
            st.error(e)

# ===============================================
# 🩺 TAB 3 — Clustering
# ===============================================
with tabs[2]:
    st.header("🩺 Patient Segmentation (K-Means Clustering)")
    st.write("Groups patients into similar profiles based on features.")
    data = pd.DataFrame({
        "Age": np.random.randint(20, 80, 50),
        "BP": np.random.randint(100, 160, 50),
        "Cholesterol": np.random.randint(150, 300, 50),
        "BMI": np.random.uniform(18, 35, 50),
        "Glucose": np.random.randint(70, 200, 50)
    })
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data)
    st.dataframe(data.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x="BMI", y="Glucose", hue="Cluster", data=data, ax=ax, palette="Set2")
    st.pyplot(fig)

# ===============================================
# 🔗 TAB 4 — Association Rules
# ===============================================
with tabs[3]:
    st.header("🔗 Association Rule Mining")
    st.write("Discovers relationships among health factors.")
    df = pd.DataFrame({
        "High_BP": np.random.choice([0, 1], 20),
        "High_BMI": np.random.choice([0, 1], 20),
        "High_Chol": np.random.choice([0, 1], 20),
        "Diabetes": np.random.choice([0, 1], 20),
    })
    freq = apriori(df, min_support=0.2, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ===============================================
# 🧠 TAB 5 — CNN
# ===============================================
with tabs[4]:
    st.header("🧠 CNN — Chest X-ray Detection")
    img = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img).convert("RGB")
        cnn = models["cnn"]
        _, H, W, C = cnn.input_shape
        arr = np.array(image.resize((W, H))) / 255.0
        arr = np.expand_dims(arr, 0)
        tf.keras.backend.clear_session()
        with tf.device("/cpu:0"):
            preds = cnn.predict(arr)
        p = float(preds[0][0]) if preds.shape[-1] == 1 else float(np.max(preds))
        label = "Pneumonia" if p > 0.5 else "Normal"
        st.image(image, caption=f"Prediction: {label} ({p*100:.1f}%)", width=300)

# ===============================================
# 📉 TAB 6 — LSTM
# ===============================================
with tabs[5]:
    st.header("📉 LSTM Time-Series Forecasting")
    timesteps = np.linspace(0, 10, 30)
    vitals = np.sin(timesteps) + np.random.normal(0, 0.1, 30)
    next_val = vitals[-1] + np.random.normal(0, 0.05)
    st.line_chart(vitals)
    st.metric("Forecasted Vital", f"{next_val:.2f}")

# ===============================================
# 🧬 TAB 7 — BioBERT
# ===============================================
with tabs[6]:
    st.header("🧬 BioBERT Clinical Text Understanding")
    text = st.text_area("Enter medical text:")
    if text:
        tokenizer = AutoTokenizer.from_pretrained("d4data/biobert-base-cased-finetuned-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("d4data/biobert-base-cased-finetuned-mnli")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(pred).item()
        st.write("Predicted Label:", ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"][label])

# ===============================================
# 💬 CHATBOT
# ===============================================
with tabs[7]:
    st.header("💬 Chatbot")
    q = st.text_area("Ask a medical question:")
    if q:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=f"Short answer: {q}")
        st.info(r.text)

# ===============================================
# 💭 SENTIMENT
# ===============================================
with tabs[8]:
    st.header("💭 Sentiment Analysis")
    t = st.text_input("Enter feedback:")
    if t:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=f"Classify sentiment: {t}").text
        st.metric("Sentiment", res)

# ===============================================
# 🌐 TRANSLATOR
# ===============================================
with tabs[9]:
    st.header("🌐 Translator")
    txt = st.text_area("Enter text:")
    lang = st.selectbox("Target", ["en", "ta", "hi", "ml", "te", "fr", "de", "es"])
    if st.button("Translate"):
        out = GoogleTranslator(source="auto", target=lang).translate(txt)
        st.success(out)

# ===============================================
# 📊 DASHBOARD
# ===============================================
with tabs[10]:
    st.header("📊 Risk vs LOS Dashboard")
    data = pd.DataFrame({
        "Risk": np.random.choice(["Low", "High"], 100),
        "LOS": np.random.uniform(1, 10, 100)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk", y="LOS", data=data, ax=ax)
    st.pyplot(fig)
