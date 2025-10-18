import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb     # 👈 this fixes ModuleNotFoundError
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import google.generativeai as genai
import os

# ----------------------
# Streamlit Setup
# ----------------------
st.set_page_config(page_title="💊 HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.write("AI-powered Health Analysis & Gemini Chatbot")

# ----------------------
# Load Models (Local Folder)
# ----------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["risk"] = joblib.load("models/risk_classifier_v1.joblib")
        st.info("✅ Risk classification model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Could not load risk model: {e}")

    try:
        models["los"] = joblib.load("models/los_regressor_v1.joblib")
        st.info("✅ LOS regression model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Could not load LOS model: {e}")

    try:
        models["cnn"] = tf.keras.models.load_model("models/keras/cnn_model_v1.h5")
        st.info("✅ CNN model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Could not load CNN model: {e}")

    try:
        models["lstm"] = tf.keras.models.load_model("models/keras/lstm_model_v1.h5")
        st.info("✅ LSTM model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Could not load LSTM model: {e}")

    try:
        models["sentiment_model"] = joblib.load("models/sentiment_model.joblib")
        models["sentiment_vectorizer"] = joblib.load("models/sentiment_vectorizer.joblib")
        st.info("✅ Sentiment model loaded.")
    except Exception as e:
        st.warning(f"⚠️ Could not load sentiment models: {e}")

    return models

models = load_models()

# ----------------------
# Gemini Config
# ----------------------
GEMINI_API_KEY = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY"))
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("⚠️ Gemini API key not found. Please set it in .streamlit/secrets.toml")

GEMINI_MODEL = "models/gemini-2.0-flash"

def gemini_call(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.header("🧠 Modules")
module = st.sidebar.selectbox(
    "Choose Module",
    [
        "🏠 Home",
        "🧬 Risk Classification",
        "🏥 LOS Regression",
        "👥 Clustering",
        "🔗 Association Rules",
        "🧠 CNN Imaging",
        "📈 LSTM Prediction",
        "🤖 Gemini Chatbot",
        "🌐 Translator",
        "❤️ Sentiment Analysis",
    ]
)

# ----------------------
# Home
# ----------------------
if module == "🏠 Home":
    st.markdown("""
    ## Welcome to HealthAI  
    Multi-model healthcare AI system featuring:
    - 🧬 Disease Risk Detection  
    - 🏥 Length of Stay Prediction  
    - 👥 Clustering & Rules  
    - 🤖 Gemini Chatbot + Translation  
    - 🧠 CNN + LSTM Diagnostics  
    """)

# ----------------------
# Risk Classification
# ----------------------
elif module == "🧬 Risk Classification":
    st.header("Disease Risk Prediction")
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

# ----------------------
# Regression
# ----------------------
elif module == "🏥 LOS Regression":
    st.header("Length of Stay Prediction")
    age = st.slider("Age", 10, 100, 50)
    sev = st.slider("Severity Level", 1, 10, 3)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    bp = st.number_input("BP", 80, 200, 120)
    glucose = st.number_input("Glucose", 50, 250, 100)
    chol = st.number_input("Cholesterol", 100, 300, 180)

    if st.button("Predict Stay"):
        try:
            X = np.array([[age, sev, bmi, bp, glucose, chol]])
            pred = models["los"].predict(X)
            st.success(f"Predicted Stay: {pred[0]:.2f} days")
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------
# Clustering
# ----------------------
elif module == "👥 Clustering":
    st.header("Patient Segmentation")
    uploaded = st.file_uploader("Upload health dataset CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        scaled = StandardScaler().fit_transform(df.select_dtypes(include=np.number))
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled)
        st.dataframe(df.head())

# ----------------------
# Association Rules
# ----------------------
elif module == "🔗 Association Rules":
    st.header("Association Rule Mining")
    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df_bool = df.astype(bool)
        freq_items = apriori(df_bool, min_support=0.2, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head())
        if st.button("Explain Rules"):
            explanation = gemini_call(f"Explain these healthcare rules: {rules.head(5)}")
            st.info(explanation)

# ----------------------
# CNN
# ----------------------
elif module == "🧠 CNN Imaging":
    st.header("CNN Medical Diagnostics")
    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded:
        img = tf.keras.utils.load_img(uploaded, target_size=(64,64))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img)/255.0, axis=0)
        pred = models["cnn"].predict(arr)
        st.success("Result: Abnormal" if pred[0][0]>0.5 else "Normal")

# ----------------------
# LSTM
# ----------------------
elif module == "📈 LSTM Prediction":
    st.header("LSTM Time Series Forecast")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        seq = np.expand_dims(df.values, axis=(0,2))
        pred = models["lstm"].predict(seq)
        st.success(f"Predicted Next Value: {pred[0][0]:.2f}")

# ----------------------
# Chatbot
# ----------------------
elif module == "🤖 Gemini Chatbot":
    st.header("AI Health Chatbot")
    lang = st.selectbox("Language", ["English","Tamil","Hindi","Malayalam"])
    q = st.text_area("Ask your medical question")
    if st.button("Get Answer"):
        st.success(gemini_call(f"Answer in {lang} within 3 bullet points: {q}"))

# ----------------------
# Translator
# ----------------------
elif module == "🌐 Translator":
    st.header("Translator")
    text = st.text_area("Enter text")
    lang = st.selectbox("Translate to", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Translate"):
        st.success(gemini_call(f"Translate to {lang}: {text}"))

# ----------------------
# Sentiment
# ----------------------
elif module == "❤️ Sentiment Analysis":
    st.header("Patient Feedback Sentiment")
    text = st.text_area("Enter feedback")
    if st.button("Analyze Sentiment"):
        result = gemini_call(f"Classify as Positive, Neutral, or Negative: {text}")
        st.write(result)
