import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
import torch
import os
import json

# =======================
# CONFIGURATIONS
# =======================
st.set_page_config(page_title="HealthAI - Smart Healthcare Assistant", page_icon="💉", layout="wide")

st.title("💊 HealthAI - Smart Healthcare Assistant")
st.write("AI-powered Smart Health Analysis and Decision Support System")

# =======================
# LOAD MODELS FROM GOOGLE DRIVE
# =======================
@st.cache_resource
def load_model_from_drive():
    import gdown

    drive_links = {
        "risk": "https://drive.google.com/uc?id=1i0Ctk06oEEKbMKig-F1W8zJ0dvWafzCx",
        "los": "https://drive.google.com/uc?id=1ziJm5oqnbpMI-Uj7jBMCFayQW8e-OHIL",
        "sentiment_model": "https://drive.google.com/uc?id=1XjNoDI6ZEngkBuql8vcEzSONp7zhwz-y",
        "sentiment_vectorizer": "https://drive.google.com/uc?id=12S8BlQGUUEQ-L-kATW2oKnJ4Ud7bRmoI",
        "cnn": "https://drive.google.com/uc?id=1QrigAH55IFbnuyhtOsuXj7Nggblhj3Wn",
        "lstm": "https://drive.google.com/uc?id=1tDQni-vP6d9UzUxF-R3hTlJVzpfc-6Hd",
    }

    os.makedirs("models", exist_ok=True)

    models = {}
    for name, link in drive_links.items():
        file_path = f"models/{name}.joblib" if "joblib" in link else f"models/{name}.h5"
        gdown.download(link, file_path, quiet=True)
        if name in ["risk", "los", "sentiment_model", "sentiment_vectorizer"]:
            models[name] = joblib.load(file_path)
        elif name == "cnn":
            models[name] = tf.keras.models.load_model(file_path)
        elif name == "lstm":
            models[name] = tf.keras.models.load_model(file_path)
    return models

models = load_model_from_drive()
st.success("✅ All models loaded successfully from Google Drive!")

# =======================
# GEMINI API CONFIG
# =======================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GENAI_API_KEY"))
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-2.0-flash"

def gemini_call(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

# =======================
# MODULE SELECTOR
# =======================
st.sidebar.header("🧠 Modules")
module = st.sidebar.radio(
    "Choose a module",
    [
        "Home",
        "Classification (Disease Risk)",
        "Regression (LOS prediction)",
        "Clustering (Patient segmentation)",
        "Association Rule Mining",
        "CNN (Imaging Diagnostics)",
        "LSTM (Time Series)",
        "Chatbot (Gemini)",
        "Translator",
        "Sentiment Analysis",
    ],
)

# =======================
# HOME
# =======================
if module == "Home":
    st.markdown("""
    ### 🩺 Welcome to HealthAI!
    HealthAI integrates Machine Learning and Generative AI to assist with:
    - Disease Risk Prediction  
    - Length of Stay Forecasting  
    - Patient Clustering  
    - Association Rule Mining  
    - CNN & LSTM Diagnostics  
    - AI Chatbot with Translation and Sentiment Analysis  

    ⚠️ *Disclaimer: This tool is for educational and informational purposes only.*
    """)

# =======================
# CLASSIFICATION
# =======================
elif module == "Classification (Disease Risk)":
    st.header("🧬 Disease Risk Classification")
    age = st.number_input("Age", 1, 100, 45)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 250, 100)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    chol = st.number_input("Cholesterol Level", 100, 300, 180)
    heart_rate = st.number_input("Heart Rate", 40, 180, 75)

    if st.button("Predict Risk"):
        try:
            features = np.array([[age, bp, glucose, bmi, chol, heart_rate]])
            pred = models["risk"].predict(features)
            st.success(f"Predicted Risk Category: {'High' if pred[0]==1 else 'Low'}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# =======================
# REGRESSION
# =======================
elif module == "Regression (LOS prediction)":
    st.header("🏥 Length of Stay (Regression)")
    age = st.slider("Age", 10, 100, 50)
    severity = st.slider("Disease Severity", 1, 10, 2)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 250, 100)
    chol = st.number_input("Cholesterol Level", 100, 300, 180)

    if st.button("Predict LOS"):
        try:
            features = np.array([[age, severity, bmi, bp, glucose, chol]])
            los = models["los"].predict(features)
            st.success(f"Predicted Length of Stay: {los[0]:.2f} days")
        except Exception as e:
            st.error(f"Regression error: {e}")

# =======================
# CLUSTERING
# =======================
elif module == "Clustering (Patient segmentation)":
    st.header("👥 Patient Segmentation")
    df = pd.read_csv("data/synthetic_health_data.csv")
    features = df[["Age", "BloodPressure", "BMI", "Glucose"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled)
    st.write(df.head())
    st.bar_chart(df["Cluster"].value_counts())

# =======================
# ASSOCIATION RULE MINING
# =======================
elif module == "Association Rule Mining":
    st.header("🔗 Association Rule Mining")
    df = pd.read_csv("data/transactions.csv")
    df_bool = df.astype(bool)
    freq_items = apriori(df_bool, min_support=0.2, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

    if st.button("🧠 Get Gemini Insight"):
        sample_rules = rules.head(5).to_string()
        insight = gemini_call(f"Explain these healthcare association rules in plain language:\n{sample_rules}")
        st.info(insight)

# =======================
# CNN
# =======================
elif module == "CNN (Imaging Diagnostics)":
    st.header("🧠 CNN - Imaging Diagnostics")
    uploaded = st.file_uploader("Upload X-ray or MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = tf.keras.utils.load_img(uploaded, target_size=(64, 64))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        pred = models["cnn"].predict(img_array)
        st.success(f"Prediction: {'Abnormal' if pred[0][0] > 0.5 else 'Normal'}")

# =======================
# LSTM
# =======================
elif module == "LSTM (Time Series)":
    st.header("📈 LSTM - Time Series Prediction")
    uploaded = st.file_uploader("Upload Time Series CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        seq = np.expand_dims(df.values, axis=(0, 2))
        pred = models["lstm"].predict(seq)
        st.success(f"Predicted next value: {pred[0][0]:.2f}")

# =======================
# CHATBOT (GEMINI)
# =======================
elif module == "Chatbot (Gemini)":
    st.header("🤖 AI Health Chatbot (Gemini)")
    lang = st.selectbox("Language", ["English", "Tamil", "Hindi", "Malayalam"])
    q = st.text_area("Ask a medical question")
    col1, col2 = st.columns(2)

    if col1.button("Get Short Answer"):
        short = gemini_call(f"Answer in {lang} within 3 short bullet points:\n{q}")
        st.success(short)

    if col2.button("Explain in Detail"):
        detailed = gemini_call(f"Explain in detail in {lang}:\n{q}")
        st.info(detailed)

# =======================
# TRANSLATOR
# =======================
elif module == "Translator":
    st.header("🌐 Translator (Gemini)")
    text = st.text_area("Enter text to translate")
    lang = st.selectbox("Translate to", ["English", "Tamil", "Hindi", "Malayalam"])
    if st.button("Translate"):
        translated = gemini_call(f"Translate this to {lang}: {text}")
        st.success(translated)

# =======================
# SENTIMENT
# =======================
elif module == "Sentiment Analysis":
    st.header("❤️ Sentiment Analysis (Gemini)")
    feedback = st.text_area("Enter patient feedback")
    if st.button("Analyze Sentiment"):
        res = gemini_call(f"""
        Analyze the sentiment (Positive, Neutral, Negative) of this text and explain why in 1 line:
        {feedback}
        """)
        st.write(res)
