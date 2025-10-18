import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import google.generativeai as genai

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.write("AI-powered Health Analysis & Gemini Chatbot")

# ----------------------
# LOAD SECRETS SAFELY
# ----------------------
GEMINI_API_KEY = st.secrets.get("GENAI_API_KEY")
FILE_IDS = {
    "risk": st.secrets.get("RISK_FILE_ID"),
    "los": st.secrets.get("LOS_FILE_ID"),
    "cnn": st.secrets.get("CNN_FILE_ID"),
    "lstm": st.secrets.get("LSTM_FILE_ID"),
    "sentiment_model": st.secrets.get("SENTIMENT_MODEL_FILE_ID"),
    "sentiment_vectorizer": st.secrets.get("SENTIMENT_VECTORIZER_FILE_ID")
}

# ----------------------
# CONFIGURE GEMINI
# ----------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    st.sidebar.success("✅ Gemini Connected")
else:
    st.sidebar.error("⚠️ Gemini API key not found in secrets.toml")

GEMINI_MODEL = "models/gemini-2.0-flash"

def gemini_call(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# ----------------------
# DOWNLOAD + LOAD MODELS
# ----------------------
os.makedirs("models", exist_ok=True)

@st.cache_resource
def download_and_load_models():
    models = {}
    for name, file_id in FILE_IDS.items():
        if not file_id:
            st.warning(f"⚠️ File ID for {name} not found in secrets.")
            continue

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        file_path = f"models/{name}.joblib" if name not in ["cnn", "lstm"] else f"models/{name}.h5"

        try:
            gdown.download(url, file_path, quiet=True)
            if name in ["risk", "los", "sentiment_model", "sentiment_vectorizer"]:
                models[name] = joblib.load(file_path)
            elif name == "cnn":
                models[name] = tf.keras.models.load_model(file_path)
            elif name == "lstm":
                models[name] = tf.keras.models.load_model(file_path)
            st.info(f"✅ Loaded {name} successfully")
        except Exception as e:
            st.warning(f"⚠️ Failed to load {name}: {e}")
            models[name] = None
    return models

models = download_and_load_models()

# ----------------------
# SIDEBAR MODULES
# ----------------------
st.sidebar.header("🧩 Modules")
module = st.sidebar.selectbox(
    "Choose a Module",
    [
        "🏠 Home",
        "🧬 Disease Risk Classification",
        "🏥 Length of Stay Prediction",
        "👥 Patient Clustering",
        "🔗 Association Rules",
        "🧠 CNN Imaging Diagnostics",
        "📈 LSTM Time Series",
        "🤖 Gemini Chatbot",
        "🌐 Translator",
        "❤️ Sentiment Analysis",
    ]
)

# ----------------------
# MODULE IMPLEMENTATIONS
# ----------------------
if module == "🏠 Home":
    st.markdown("""
    ### Welcome to HealthAI  
    This AI system integrates Gemini + ML models:
    - 🧬 Disease Risk Detection  
    - 🏥 Length of Stay Regression  
    - 👥 Clustering & Rule Mining  
    - 🧠 CNN + LSTM Models  
    - 🤖 Chatbot + Translator + Sentiment  
    """)

elif module == "🧬 Disease Risk Classification":
    st.header("🧬 Predict Disease Risk")
    age = st.number_input("Age", 10, 100, 45)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    glucose = st.number_input("Glucose", 50, 250, 100)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    chol = st.number_input("Cholesterol", 100, 300, 180)
    heart = st.number_input("Heart Rate", 40, 180, 75)

    if st.button("Predict Risk"):
        try:
            X = np.array([[age, bp, glucose, bmi, chol, heart]])
            pred = models["risk"].predict(X)
            st.success(f"Predicted Risk: {'High' if pred[0]==1 else 'Low'}")
        except Exception as e:
            st.error(f"Error: {e}")

elif module == "🏥 Length of Stay Prediction":
    st.header("🏥 Predict Hospital Stay Duration")
    age = st.slider("Age", 10, 100, 40)
    severity = st.slider("Severity", 1, 10, 3)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    bp = st.number_input("BP", 80, 200, 120)
    glucose = st.number_input("Glucose", 50, 250, 100)
    chol = st.number_input("Cholesterol", 100, 300, 180)

    if st.button("Predict LOS"):
        try:
            X = np.array([[age, severity, bmi, bp, glucose, chol]])
            los = models["los"].predict(X)
            st.success(f"Predicted Stay: {los[0]:.2f} days")
        except Exception as e:
            st.error(f"Error: {e}")

elif module == "👥 Patient Clustering":
    st.header("Patient Clustering")
    uploaded = st.file_uploader("Upload patient data (CSV)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        scaled = StandardScaler().fit_transform(df.select_dtypes(include=np.number))
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled)
        st.dataframe(df.head())

elif module == "🔗 Association Rules":
    st.header("Association Rule Mining")
    uploaded = st.file_uploader("Upload transactions.csv", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df_bool = df.astype(bool)
        freq = apriori(df_bool, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head())
        if st.button("Explain Rules"):
            st.info(gemini_call(f"Explain these medical association rules:\n{rules.head(5)}"))

elif module == "🧠 CNN Imaging Diagnostics":
    st.header("CNN Image Prediction")
    uploaded = st.file_uploader("Upload MRI/X-ray Image", type=["jpg","jpeg","png"])
    if uploaded:
        img = tf.keras.utils.load_img(uploaded, target_size=(64,64))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img)/255.0, axis=0)
        pred = models["cnn"].predict(arr)
        st.success("Result: Abnormal" if pred[0][0]>0.5 else "Normal")

elif module == "📈 LSTM Time Series":
    st.header("LSTM Time Series Prediction")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        seq = np.expand_dims(df.values, axis=(0,2))
        pred = models["lstm"].predict(seq)
        st.success(f"Next Value Prediction: {pred[0][0]:.2f}")

elif module == "🤖 Gemini Chatbot":
    st.header("Gemini Chatbot")
    lang = st.selectbox("Language", ["English","Tamil","Hindi","Malayalam"])
    q = st.text_area("Ask your question")
    if st.button("Ask Gemini"):
        st.success(gemini_call(f"Answer in {lang} briefly: {q}"))

elif module == "🌐 Translator":
    st.header("Translator")
    txt = st.text_area("Enter text to translate")
    lang = st.selectbox("Translate to", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Translate"):
        st.success(gemini_call(f"Translate this to {lang}: {txt}"))

elif module == "❤️ Sentiment Analysis":
    st.header("Sentiment Analysis")
    feedback = st.text_area("Enter feedback text")
    if st.button("Analyze"):
        st.info(gemini_call(f"Classify this sentiment (Positive, Neutral, or Negative): {feedback}"))
