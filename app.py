# app.py — HealthAI (Light Streamlit Version)

import streamlit as st
import numpy as np
import pandas as pd
import os, joblib, gdown, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="HealthAI - Smart Healthcare Assistant", page_icon="💊", layout="wide")

# Fade-out success message
def fade_message(msg):
    st.markdown(
        f"""
        <div style='background-color:#eafbea;padding:8px;border-radius:6px;'>✅ {msg}</div>
        <script>
        setTimeout(() => {{
            let el = window.parent.document.querySelectorAll('section[data-testid="stNotification"]');
            el.forEach(e => e.remove());
        }}, 2500);
        </script>
        """, unsafe_allow_html=True
    )

# Download from Drive using file ID (no public links)
def download_from_drive(name, fid, path):
    if not fid: return None
    if os.path.exists(path): return path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={fid}", path, quiet=True)
        return path if os.path.exists(path) else None
    except Exception as e:
        st.warning(f"⚠️ {name} download failed: {e}")
        return None

# Load models (cached)
@st.cache_resource
def load_models():
    models = {}
    ids = st.secrets
    files = {
        "risk": (ids.get("DRIVE_FILE_RISK", ""), "models/risk.joblib"),
        "los": (ids.get("DRIVE_FILE_LOS", ""), "models/los.joblib"),
        "cnn": (ids.get("DRIVE_FILE_CNN_H5", ""), "models/cnn_model.h5"),
        "lstm": (ids.get("DRIVE_FILE_LSTM_H5", ""), "models/lstm_model.h5"),
        "sentiment_model": (ids.get("DRIVE_FILE_SENTIMENT_MODEL", ""), "models/sentiment_model.joblib"),
        "sentiment_vectorizer": (ids.get("DRIVE_FILE_SENTIMENT_VEC", ""), "models/sentiment_vectorizer.joblib"),
    }
    for name, (fid, path) in files.items():
        if not fid:
            continue
        local = download_from_drive(name, fid, path)
        if not local:
            continue
        try:
            if name in ["risk", "los", "sentiment_model", "sentiment_vectorizer"]:
                models[name] = joblib.load(local)
            else:
                models[name] = tf.keras.models.load_model(local)
            fade_message(f"{name} loaded")
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")
    return models

models = load_models()

# Sidebar navigation
st.sidebar.title("💊 HealthAI")
page = st.sidebar.radio("Choose Module", [
    "🏠 Home",
    "🧬 Classification",
    "🏥 Regression",
    "👥 Clustering",
    "🔗 Association Rules",
    "🧩 CNN",
    "📈 LSTM",
    "💬 Chatbot",
    "🌐 Translator",
    "❤️ Sentiment"
])

# Home
if page == "🏠 Home":
    st.title("💊 HealthAI - Smart Healthcare Assistant")
    st.markdown("AI system combining ML, DL, and Gemini AI for healthcare analytics.")
    fig, ax = plt.subplots()
    ax.bar(["Risk", "LOS", "CNN", "LSTM"], [0.9, 0.85, 0.95, 0.9])
    ax.set_ylabel("Model Accuracy (example)")
    st.pyplot(fig)

# Risk Prediction
elif page == "🧬 Classification":
    st.header("🧬 Disease Risk Prediction")
    a = st.slider("Age", 10, 100, 45)
    b = st.slider("BP", 80, 200, 120)
    g = st.slider("Glucose", 50, 250, 100)
    bmi = st.slider("BMI", 10.0, 40.0, 25.0)
    if st.button("Predict"):
        try:
            X = np.array([[a, b, g, bmi]])
            p = models["risk"].predict(X)
            st.success("High Risk 🚨" if p[0] == 1 else "Low Risk ✅")
        except Exception as e:
            st.error(e)

# Regression (LOS)
elif page == "🏥 Regression":
    st.header("🏥 Hospital Stay Prediction")
    a = st.number_input("Age", 10, 100, 50)
    sev = st.slider("Severity", 1, 10, 5)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
    if st.button("Predict Stay"):
        try:
            X = np.array([[a, sev, bmi]])
            res = models["los"].predict(X)
            st.success(f"Expected Stay: {res[0]:.2f} days")
        except Exception as e:
            st.error(e)

# Clustering
elif page == "👥 Clustering":
    st.header("👥 Patient Segmentation")
    f = st.file_uploader("Upload patient dataset", type="csv")
    if f:
        df = pd.read_csv(f)
        X = df.select_dtypes(include=np.number)
        km = KMeans(n_clusters=3, random_state=0)
        df["Cluster"] = km.fit_predict(X)
        pca = PCA(n_components=2)
        pts = pca.fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(pts[:, 0], pts[:, 1], c=df["Cluster"], cmap="tab10")
        st.pyplot(fig)

# Association Rules
elif page == "🔗 Association Rules":
    st.header("🔗 Apriori Rule Mining")
    up = st.file_uploader("Upload transactions.csv", type="csv")
    if up:
        df = pd.read_csv(up)
        from mlxtend.preprocessing import TransactionEncoder
        basket = df.iloc[:, 0].astype(str).apply(lambda x: x.split(","))
        te = TransactionEncoder()
        arr = te.fit(basket).transform(basket)
        df_enc = pd.DataFrame(arr, columns=te.columns_)
        freq = apriori(df_enc, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head(10))

# CNN
elif page == "🧩 CNN":
    st.header("🧩 Chest X-ray Classifier")
    img = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(image) / 255.0, 0)
        try:
            pred = models["cnn"].predict(arr)
            label = "PNEUMONIA ⚠️" if np.argmax(pred) == 1 else "NORMAL ✅"
            st.image(image, caption=f"Prediction: {label}", width=300)
        except Exception as e:
            st.error(e)

# LSTM
elif page == "📈 LSTM":
    st.header("📈 LSTM Vitals Forecasting")
    f = st.file_uploader("Upload lstm_input_full.csv", type=["csv"])
    if f:
        df = pd.read_csv(f)
        st.line_chart(df.set_index("timestamp")[["heart_rate", "spo2", "resp_rate"]])
        seq = np.expand_dims(df[["heart_rate", "spo2", "resp_rate"]].values[:10], axis=0)
        pred = models["lstm"].predict(seq)
        st.success(f"Predicted next value: {float(pred[0][0]):.2f}")

# Chatbot
elif page == "💬 Chatbot":
    st.header("💬 Gemini Health Chatbot")
    q = st.text_input("Ask your question")
    if st.button("Ask"):
        import google.generativeai as genai
        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
        try:
            resp = genai.generate_content(q)
            st.write(resp.text)
        except Exception as e:
            st.error(e)

# Translator
elif page == "🌐 Translator":
    st.header("🌐 Translator")
    text = st.text_area("Enter text")
    lang = st.selectbox("Language", ["Tamil", "Hindi", "Malayalam", "English"])
    if st.button("Translate"):
        from googletrans import Translator
        tr = Translator()
        langs = {"Tamil": "ta", "Hindi": "hi", "Malayalam": "ml", "English": "en"}
        res = tr.translate(text, dest=langs[lang])
        st.success(res.text)

# Sentiment
elif page == "❤️ Sentiment":
    st.header("❤️ Patient Feedback Sentiment")
    text = st.text_area("Enter feedback")
    if st.button("Analyze"):
        try:
            vec = models["sentiment_vectorizer"].transform([text])
            y = models["sentiment_model"].predict(vec)[0]
            st.success("Positive 😀" if y == 1 else "Negative 😞")
        except Exception as e:
            st.error(e)

st.caption("© 2025 HealthAI | Built with Streamlit + TensorFlow + Gemini")
