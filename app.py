import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image

# ========================
# Streamlit configuration
# ========================
st.set_page_config(page_title="HealthAI - Smart Healthcare Assistant", page_icon="💊", layout="wide")

# ========================
# Fade message (auto erase)
# ========================
def show_fade_success(message, delay=3000):
    st.markdown(f"""
    <div style='background-color:#e8f9f0;padding:10px;border-radius:8px;margin-bottom:8px;'>
    ✅ {message}
    </div>
    <script>
    setTimeout(() => {{
        const msg = window.parent.document.querySelectorAll('section[data-testid="stNotification"]');
        msg.forEach(el => el.remove());
    }}, {delay});
    </script>
    """, unsafe_allow_html=True)

# ========================
# Download from Drive using secrets (no public link)
# ========================
def download_model(name, file_id, path):
    if not os.path.exists("models"):
        os.makedirs("models")
    if os.path.exists(path):
        return path
    if not file_id:
        return None
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=True, fuzzy=True)
        return path if os.path.exists(path) else None
    except Exception as e:
        st.warning(f"⚠️ {name} download failed: {e}")
        return None

# ========================
# Load Models
# ========================
@st.cache_resource
def load_models():
    models = {}
    secrets = st.secrets

    files = {
        "risk": (secrets.get("DRIVE_FILE_RISK", ""), "models/risk.joblib"),
        "los": (secrets.get("DRIVE_FILE_LOS", ""), "models/los.joblib"),
        "cnn": (secrets.get("DRIVE_FILE_CNN_H5", ""), "models/cnn_model.h5"),
        "lstm": (secrets.get("DRIVE_FILE_LSTM_H5", ""), "models/lstm_model.h5"),
        "sentiment_model": (secrets.get("DRIVE_FILE_SENTIMENT_MODEL", ""), "models/sentiment_model.joblib"),
        "sentiment_vectorizer": (secrets.get("DRIVE_FILE_SENTIMENT_VEC", ""), "models/sentiment_vectorizer.joblib"),
    }

    for name, (fid, path) in files.items():
        if not fid:
            continue
        fpath = download_model(name, fid, path)
        try:
            if name in ["risk", "los", "sentiment_model", "sentiment_vectorizer"]:
                models[name] = joblib.load(fpath)
            else:
                models[name] = tf.keras.models.load_model(fpath)
            show_fade_success(f"Loaded {name} successfully")
        except Exception as e:
            st.warning(f"⚠️ {name} failed to load: {e}")
    return models

models = load_models()

# ========================
# Main Title
# ========================
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("AI-powered Health Analysis & Gemini Chatbot")

# Sidebar Navigation
module = st.sidebar.radio("Modules", [
    "🏠 Home",
    "🧬 Classification (Disease Risk)",
    "🏥 Regression (Length of Stay)",
    "👥 Clustering (Patient Segmentation)",
    "🔗 Association Rule Mining",
    "🧩 CNN (Imaging Diagnostics)",
    "📈 LSTM (Time Series)",
    "💬 Chatbot (Gemini)",
    "🌐 Translator",
    "❤️ Sentiment Analysis"
])

# =====================================
# Home
# =====================================
if module == "🏠 Home":
    st.markdown("""
    ### Welcome to HealthAI 👋  
    This intelligent system integrates:
    - 🧬 Disease Risk Classification  
    - 🏥 Hospital Stay Regression  
    - 👥 Patient Clustering  
    - 🔗 Association Rule Mining  
    - 🧩 CNN & LSTM Models  
    - 💬 Gemini Chatbot + Translation  
    - ❤️ Sentiment Analysis  
    """)

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Risk", "LOS", "CNN", "LSTM"], [0.9, 0.8, 0.95, 0.92])
    ax.set_title("Model Confidence Overview")
    ax.set_ylabel("Accuracy (Example)")
    st.pyplot(fig)

# =====================================
# Classification
# =====================================
elif module == "🧬 Classification (Disease Risk)":
    st.header("🧬 Disease Risk Classification")
    age = st.number_input("Age", 10, 100, 40)
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 250, 100)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)

    if st.button("Predict Risk"):
        try:
            X = np.array([[age, bp, glucose, bmi]])
            pred = models["risk"].predict(X)
            st.success(f"Prediction: {'High Risk' if pred[0]==1 else 'Low Risk'}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =====================================
# Regression (LOS)
# =====================================
elif module == "🏥 Regression (Length of Stay)":
    st.header("🏥 Predict Hospital Stay (Regression)")
    age = st.slider("Age", 10, 100, 50)
    severity = st.slider("Disease Severity", 1, 10, 4)
    bmi = st.number_input("BMI", 10.0, 40.0, 25.0)

    if st.button("Predict LOS"):
        try:
            X = np.array([[age, severity, bmi]])
            result = models["los"].predict(X)
            st.success(f"Predicted Hospital Stay: {float(result[0]):.2f} days")
        except Exception as e:
            st.error(f"Error: {e}")

# =====================================
# Clustering (KMeans)
# =====================================
elif module == "👥 Clustering (Patient Segmentation)":
    st.header("👥 Patient Segmentation with KMeans")
    file = st.file_uploader("Upload Patient Dataset (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        n_clusters = st.slider("Select Clusters", 2, 10, 3)
        X = df.select_dtypes(include=np.number).dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(proj[:, 0], proj[:, 1], c=df["Cluster"], cmap="tab10")
        ax.set_title("Patient Clustering Projection")
        st.pyplot(fig)

# =====================================
# Association Rules
# =====================================
elif module == "🔗 Association Rule Mining":
    st.header("🔗 Association Rule Mining (Apriori)")
    trans = st.file_uploader("Upload transactions.csv", type=["csv"])
    if trans:
        df = pd.read_csv(trans)
        from mlxtend.preprocessing import TransactionEncoder
        basket = df.iloc[:, 0].astype(str).apply(lambda x: x.split(","))
        te = TransactionEncoder()
        te_ary = te.fit(basket).transform(basket)
        df_ohe = pd.DataFrame(te_ary, columns=te.columns_)
        frequent = apriori(df_ohe, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head(10))

# =====================================
# CNN
# =====================================
elif module == "🧩 CNN (Imaging Diagnostics)":
    st.header("🧩 CNN Medical Image Classifier (Normal/Pneumonia)")
    img = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    if img:
        try:
            image = Image.open(img).convert("RGB").resize((224,224))
            arr = np.expand_dims(np.array(image)/255.0, axis=0)
            preds = models["cnn"].predict(arr)
            label = "PNEUMONIA" if np.argmax(preds) == 1 else "NORMAL"
            st.image(image, caption=f"Prediction: {label}", width=300)
            st.success(f"Result: {label} (Confidence: {np.max(preds):.2f})")
        except Exception as e:
            st.error(f"CNN Prediction Error: {e}")

# =====================================
# LSTM
# =====================================
elif module == "📈 LSTM (Time Series)":
    st.header("📈 LSTM Time Series Forecasting")
    file = st.file_uploader("Upload lstm_input_full.csv", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.line_chart(df.set_index("timestamp")["heart_rate"])
        st.line_chart(df.set_index("timestamp")[["spo2", "resp_rate"]])

        seq = np.expand_dims(df[["heart_rate","spo2","resp_rate"]].values[:10], axis=0)
        pred = models["lstm"].predict(seq)
        st.success(f"Predicted next value (demo): {float(pred[0][0]):.2f}")

# =====================================
# Chatbot
# =====================================
elif module == "💬 Chatbot (Gemini)":
    st.header("💬 Health Chatbot (Gemini)")
    question = st.text_input("Ask a health question:")
    if st.button("Ask"):
        st.info(f"Gemini would respond here for: {question}")

# =====================================
# Translator
# =====================================
elif module == "🌐 Translator":
    st.header("🌐 Translator (Gemini)")
    st.info("Gemini-based translation feature placeholder.")

# =====================================
# Sentiment
# =====================================
elif module == "❤️ Sentiment Analysis":
    st.header("❤️ Patient Feedback Sentiment")
    text = st.text_area("Enter patient feedback")
    if st.button("Analyze Sentiment"):
        try:
            vec = models["sentiment_vectorizer"].transform([text])
            pred = models["sentiment_model"].predict(vec)[0]
            st.success(f"Predicted Sentiment: {'Positive 😀' if pred==1 else 'Negative 😞'}")
        except Exception as e:
            st.error(f"Error: {e}")
