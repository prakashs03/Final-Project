import os
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from googletrans import Translator
import tensorflow as tf
import google.generativeai as genai
import xgboost as xgb  # ✅ Added for XGBoost support

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="💊 HealthAI - Smart Healthcare Assistant", page_icon="💊", layout="wide")
st.markdown("<h1 style='text-align:center;'>💊 HealthAI - Smart Healthcare Assistant</h1>", unsafe_allow_html=True)
st.caption("AI-powered Health Analysis, Imaging (CNN), Time-series (LSTM), Chatbot (Gemini), Translator & Sentiment")

# ----------------------------------------------------------
# Google Drive Download Helper
# ----------------------------------------------------------
@st.cache_resource
def download_from_drive(file_id, output_path):
    """Download a file from Google Drive using its file ID."""
    if not file_id:
        return None
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path if os.path.exists(output_path) else None


# ----------------------------------------------------------
# Model Loader
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    messages = []
    drive_keys = {
        "risk": ("RISK_FILE_ID", "models/risk_model.joblib"),
        "los": ("LOS_FILE_ID", "models/los_model.joblib"),
        "cnn": ("CNN_FILE_ID", "models/cnn_model.h5"),
        "lstm": ("LSTM_FILE_ID", "models/lstm_model.h5"),
        "sentiment_model": ("SENTIMENT_MODEL_FILE_ID", "models/sentiment_model.joblib"),
        "sentiment_vectorizer": ("SENTIMENT_VECTORIZER_FILE_ID", "models/sentiment_vectorizer.joblib"),
    }

    for key, (secret_key, path) in drive_keys.items():
        file_id = st.secrets.get(secret_key)
        try:
            if file_id:
                local_path = download_from_drive(file_id, path)
                if local_path:
                    if local_path.endswith(".joblib"):
                        models[key] = joblib.load(local_path)
                    else:
                        models[key] = tf.keras.models.load_model(local_path, compile=False)
                    messages.append(f"✅ Loaded {key} successfully.")
                else:
                    messages.append(f"⚠️ Could not load {key} (download failed).")
            else:
                messages.append(f"⚠️ {key} file ID missing in secrets.")
        except Exception as e:
            messages.append(f"⚠️ Error loading {key}: {e}")

    return models, messages


models, load_messages = load_models()
for msg in load_messages:
    st.info(msg)
st.markdown("<script>setTimeout(()=>{document.querySelectorAll('.stAlert').forEach(e=>e.remove());},3500)</script>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Grad-CAM for CNN Visualization
# ----------------------------------------------------------
def grad_cam(image_array, model):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer = layer.name
                break
        if not last_conv_layer:
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_heatmap(original, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(original.size)
    heatmap_color = plt.cm.jet(np.array(heatmap_img) / 255.0)[:, :, :3]
    heatmap_color = Image.fromarray((heatmap_color * 255).astype("uint8"))
    return Image.blend(original.convert("RGBA"), heatmap_color.convert("RGBA"), alpha)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
st.sidebar.title("🧭 Modules")
page = st.sidebar.radio("Select a module", [
    "🏠 Home",
    "🧬 Disease Risk Prediction",
    "🏥 Length of Stay Prediction",
    "👥 Patient Clustering",
    "🔗 Association Rule Mining",
    "🧠 CNN Imaging Diagnostics",
    "📊 LSTM Vitals Forecast",
    "💬 Gemini Chatbot",
    "🌐 Translator",
    "❤️ Sentiment Analysis"
])

# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

if page == "🏠 Home":
    st.header("Welcome to HealthAI 👋")
    st.markdown("""
    **HealthAI** integrates multiple AI-powered modules:
    - 🧬 Disease Risk Prediction (XGBoost)
    - 🏥 Length of Stay (Regression)
    - 🧠 CNN Imaging Diagnostics (Grad-CAM Visualization)
    - 📊 LSTM Time-Series Forecasting
    - 👥 Clustering & 🔗 Association Rule Mining
    - 💬 Gemini Chatbot & 🌐 Translator
    - ❤️ Sentiment Analysis
    """)

elif page == "🧬 Disease Risk Prediction":
    st.header("🧬 Disease Risk Prediction (XGBoost)")
    age = st.slider("Age", 18, 100, 45)
    bp = st.slider("Blood Pressure", 80, 200, 120)
    glucose = st.slider("Glucose Level", 50, 250, 100)
    bmi = st.slider("BMI", 10.0, 40.0, 24.5)
    if st.button("Predict Risk"):
        model = models.get("risk")
        if model:
            X = np.array([[age, bp, glucose, bmi]])
            pred = model.predict(X)
            result = "🚨 High Risk" if pred[0] > 0.5 else "✅ Low Risk"
            st.success(result)
        else:
            st.error("Risk model not loaded or XGBoost missing.")

elif page == "🏥 Length of Stay Prediction":
    st.header("🏥 Predict Length of Stay (LOS)")
    age = st.number_input("Age", 0, 120, 50)
    severity = st.slider("Severity Level", 1, 10, 3)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    if st.button("Predict LOS"):
        model = models.get("los")
        if model:
            X = np.array([[age, severity, bmi]])
            pred = model.predict(X)
            st.success(f"🕓 Estimated Stay: {float(pred[0]):.2f} days")
        else:
            st.error("LOS model not loaded or XGBoost missing.")

elif page == "🧠 CNN Imaging Diagnostics":
    st.header("🧠 CNN Imaging Diagnostics (Grad-CAM)")
    img_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
    model = models.get("cnn")
    if img_file and model:
        image = Image.open(img_file).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(image) / 255.0, 0)
        pred = model.predict(arr)
        label = "⚠️ Pneumonia Detected" if np.argmax(pred) == 1 else "✅ Normal"
        st.image(image, caption=f"Prediction: {label}", width=300)
        heatmap = grad_cam(arr, model)
        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption="Grad-CAM Heatmap", width=300)
    elif not model:
        st.error("CNN model not loaded.")

elif page == "📊 LSTM Vitals Forecast":
    st.header("📊 LSTM Vitals Forecast (Predicted vs Actual Plot)")
    df = pd.read_csv("data/vitals.csv") if os.path.exists("data/vitals.csv") else pd.DataFrame({
        "Time": np.arange(20), "HeartRate": np.random.randint(60, 100, 20)
    })
    st.line_chart(df.set_index("Time"))
    model = models.get("lstm")
    if model is not None:
        seq = np.expand_dims(df["HeartRate"].values[-10:], axis=(0, 2))
        pred = model.predict(seq)
        fig, ax = plt.subplots()
        ax.plot(df["HeartRate"], label="Actual HeartRate")
        ax.scatter(len(df), pred[0][0], color="red", label="Predicted Next")
        ax.legend()
        st.pyplot(fig)
        st.success(f"Predicted Next Heart Rate: {pred[0][0]:.2f}")

elif page == "💬 Gemini Chatbot":
    st.header("💬 Gemini Chatbot")
    query = st.text_input("Ask your medical question:")
    if st.button("Ask Gemini"):
        api_key = st.secrets.get("GENAI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            response = genai.generate_content(f"You are a healthcare assistant. {query}")
            st.write(response.text)
        else:
            st.error("Gemini API key not found in secrets.")

elif page == "🌐 Translator":
    st.header("🌐 Translator")
    text = st.text_area("Enter text to translate")
    lang = st.selectbox("Translate to", ["English", "Tamil", "Hindi", "Malayalam"])
    if st.button("Translate"):
        trans = Translator()
        dest = {"English": "en", "Tamil": "ta", "Hindi": "hi", "Malayalam": "ml"}[lang]
        result = trans.translate(text, dest=dest)
        st.success(result.text)

elif page == "❤️ Sentiment Analysis":
    st.header("❤️ Sentiment Analysis (Patient Feedback)")
    text = st.text_area("Enter patient feedback")
    if st.button("Analyze Sentiment"):
        model = models.get("sentiment_model")
        vectorizer = models.get("sentiment_vectorizer")
        if model and vectorizer:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            st.success("😀 Positive" if pred == 1 else "😞 Negative")
        else:
            st.error("Sentiment model or vectorizer not loaded.")
