import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import google.generativeai as genai

# =======================
# PAGE SETUP
# =======================
st.set_page_config(page_title="HealthAI", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")

st.write("AI-powered Health Analysis & Gemini Chatbot")

# =======================
# GOOGLE DRIVE LINKS
# =======================
DRIVE_FILES = {
    "risk": "1i0Ctk06oEEKbMKig-F1W8zJ0dvWafzCx",     # risk_classifier_v1.joblib
    "los":  "1ziJm5oqnbpMI-Uj7jBMCFayQW8e-OHIL",     # los_regressor_v1.joblib
    "cnn_h5": "1QrigAH55IFbnuyhtOsuXj7Nggblhj3Wn",   # cnn_model_v1.h5
    "lstm_h5": "1tDQni-vP6d9UzUxF-R3hTlJVzpfc-6Hd",  # lstm_model_v1.h5
    "sentiment_model":"1XjNoDI6ZEngkBuql8vcEzSONp7zhwz-y",
    "sentiment_vectorizer":"12S8BlQGUUEQ-L-kATW2oKnJ4Ud7bRmoI"
}

os.makedirs("models", exist_ok=True)

# =======================
# MODEL DOWNLOAD + LOAD
# =======================
@st.cache_resource
def download_and_load_models():
    models = {}
    for name, file_id in DRIVE_FILES.items():
        file_path = f"models/{name}.joblib" if 'joblib' in name else f"models/{name}.h5"
        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            gdown.download(url, file_path, quiet=True)
            if name in ["risk", "los", "sentiment_model", "sentiment_vectorizer"]:
                models[name] = joblib.load(file_path)
            elif name == "cnn_h5":
                models["cnn"] = tf.keras.models.load_model(file_path)
            elif name == "lstm_h5":
                models["lstm"] = tf.keras.models.load_model(file_path)
            st.info(f"✅ Loaded: {name}")
        except Exception as e:
            st.warning(f"⚠️ Skipped {name}: {e}")
            models[name] = None
    return models

models = download_and_load_models()

# =======================
# GEMINI CONFIG
# =======================
GEMINI_API_KEY = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY"))
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ Gemini API key not found in secrets.toml")

GEMINI_MODEL = "models/gemini-2.0-flash"

def gemini_call(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

# =======================
# SIDEBAR MODULES
# =======================
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

# =======================
# MODULES
# =======================

# HOME
if module == "🏠 Home":
    st.subheader("Welcome to HealthAI 👋")
    st.markdown("""
    HealthAI integrates ML + Gemini AI for healthcare:
    - 🧬 Disease Risk Classification  
    - 🏥 Length of Stay Regression  
    - 👥 Clustering & Association Mining  
    - 🤖 Chatbot with Translation + Sentiment  
    - 🧠 CNN & LSTM Models  
    """)

# RISK CLASSIFICATION
elif module == "🧬 Disease Risk Classification":
    st.header("Disease Risk Prediction")
    age = st.number_input("Age", 10, 100, 45)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    glucose = st.number_input("Glucose", 50, 250, 100)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    chol = st.number_input("Cholesterol", 100, 300, 180)
    heart = st.number_input("Heart Rate", 40, 180, 75)

    if st.button("Predict Risk"):
        try:
            features = np.array([[age, bp, glucose, bmi, chol, heart]])
            pred = models["risk"].predict(features)
            label = "High Risk" if pred[0] == 1 else "Low Risk"
            st.success(f"Predicted Risk: {label}")
        except Exception as e:
            st.error(f"Error: {e}")

# REGRESSION
elif module == "🏥 Length of Stay Prediction":
    st.header("Predict Hospital Stay (LOS)")
    age = st.slider("Age", 10, 100, 40)
    severity = st.slider("Severity Level", 1, 10, 3)
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

# CLUSTERING
elif module == "👥 Patient Clustering":
    st.header("Patient Segmentation")
    uploaded = st.file_uploader("Upload health dataset CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        features = df.select_dtypes(include=np.number)
        scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled)
        st.dataframe(df.head())
        st.bar_chart(df["Cluster"].value_counts())

# ASSOCIATION RULES
elif module == "🔗 Association Rules":
    st.header("Association Rule Mining")
    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df_bool = df.astype(bool)
        freq_items = apriori(df_bool, min_support=0.2, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head())

        if st.button("Explain Rules with Gemini"):
            sample = rules.head(5).to_string()
            insight = gemini_call(f"Explain these medical association rules:\n{sample}")
            st.info(insight)

# CNN
elif module == "🧠 CNN Imaging Diagnostics":
    st.header("CNN Medical Image Classification")
    uploaded = st.file_uploader("Upload MRI/X-ray Image", type=["jpg","jpeg","png"])
    if uploaded:
        img = tf.keras.utils.load_img(uploaded, target_size=(64,64))
        img_arr = tf.keras.utils.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)/255.0
        pred = models["cnn"].predict(img_arr)
        st.success(f"Prediction: {'Abnormal' if pred[0][0]>0.5 else 'Normal'}")

# LSTM
elif module == "📈 LSTM Time Series":
    st.header("LSTM Time Series Prediction")
    uploaded = st.file_uploader("Upload CSV (time series)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        seq = np.expand_dims(df.values, axis=(0,2))
        pred = models["lstm"].predict(seq)
        st.success(f"Next value prediction: {pred[0][0]:.2f}")

# CHATBOT
elif module == "🤖 Gemini Chatbot":
    st.header("AI Chatbot")
    lang = st.selectbox("Language", ["English", "Tamil", "Hindi", "Malayalam"])
    q = st.text_area("Ask your question")
    col1, col2 = st.columns(2)
    if col1.button("Short Answer"):
        st.success(gemini_call(f"Answer in {lang} briefly (2-3 bullets): {q}"))
    if col2.button("Detailed Explanation"):
        st.info(gemini_call(f"Explain in {lang} in detail: {q}"))

# TRANSLATOR
elif module == "🌐 Translator":
    st.header("Translator (Gemini)")
    txt = st.text_area("Enter text")
    lang = st.selectbox("Translate to", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Translate"):
        st.success(gemini_call(f"Translate to {lang}: {txt}"))

# SENTIMENT
elif module == "❤️ Sentiment Analysis":
    st.header("Patient Feedback Sentiment")
    feedback = st.text_area("Enter feedback")
    if st.button("Analyze Sentiment"):
        result = gemini_call(f"Classify this sentiment (Positive, Neutral, Negative) with 1-line reason:\n{feedback}")
        st.write(result)
