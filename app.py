# app.py
"""
HealthAI - Streamlit app that:
- loads models from Google Drive links stored in st.secrets["DRIVE"]
- if missing, creates small fallback models so the UI never crashes
- supports: Risk classification, LOS regression, Clustering, Association rules,
  CNN imaging (with Grad-CAM best-effort), LSTM time-series, Gemini chatbot,
  Translator (googletrans) and Sentiment.
"""

import os
import io
import time
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# try import tensorflow (Keras). If absent, create pure-numpy fallbacks.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as keras_models
except Exception:
    tf = None
    keras_models = None

# optional: google generative ai (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# translator
try:
    from googletrans import Translator as GT_Translator
except Exception:
    GT_Translator = None

# Page config
st.set_page_config(page_title="HealthAI - Smart Healthcare Assistant",
                   layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")
st.caption("Multimodal healthcare demo — risk, LOS, clustering, association, CNN, LSTM, chatbot, translation, sentiment")

# -------------------------
# Utility: Drive downloader
# -------------------------
def drive_id_from_link(link: str) -> str:
    if not link: return None
    if "uc?id=" in link:
        return link.split("uc?id=")[1].split("&")[0]
    if "/d/" in link:
        return link.split("/d/")[1].split("/")[0]
    return link

@st.cache_resource
def download_file_from_drive(secret_key: str, out_name: str):
    """Download file from st.secrets["DRIVE"][secret_key] -> models/out_name"""
    try:
        link = st.secrets["DRIVE"].get(secret_key)
    except Exception:
        link = None
    if not link:
        return None
    fid = drive_id_from_link(link)
    if not fid:
        return None
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", out_name)
    if os.path.exists(out_path):
        return out_path
    url = f"https://drive.google.com/uc?id={fid}"
    try:
        gdown.download(url, out_path, quiet=True)
        if os.path.exists(out_path):
            return out_path
    except Exception:
        return None
    return None

# -------------------------
# Fallback model builders
# -------------------------
def build_fallback_risk_model():
    # small logistic regression trained on synthetic features (age,bp,glucose,bmi)
    X = np.random.normal(size=(200,4))
    y = (X[:,0] * 0.03 + X[:,2]*0.7 + np.random.randn(200)*0.1) > 0.5
    clf = LogisticRegression()
    clf.fit(X, y.astype(int))
    return clf

def build_fallback_los_model():
    X = np.random.normal(size=(200,3))
    y = (2 + X[:,0]*0.1 + X[:,1]*0.5 + X[:,2]*0.05 + np.random.randn(200)*0.2)
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

def build_fallback_sentiment():
    # tiny text classifier pipeline
    texts = ["good service", "very satisfied", "bad experience", "not good", "excellent", "terrible"]
    y = [1,1,0,0,1,0]
    pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=200))
    pipe.fit(texts, y)
    return pipe

def build_small_cnn(input_shape=(224,224,3), n_classes=2):
    if tf is None:
        return None
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8,3,activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(16,3,activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid" if n_classes==1 else "softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy" if n_classes==1 else "categorical_crossentropy")
    return model

def build_small_lstm(input_steps=10, features=1):
    if tf is None:
        return None
    inp = tf.keras.Input(shape=(input_steps, features))
    x = tf.keras.layers.LSTM(16)(inp)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Load models (drive or fallback)
# -------------------------
@st.cache_resource
def load_models_with_fallback():
    loaded = {}
    msgs = []
    # mapping of secret keys -> (local filename, loader type)
    map_ = {
        "risk": ("risk_classifier_v1.joblib", "joblib"),
        "los": ("los_regressor_v1.joblib", "joblib"),
        "cnn_h5": ("cnn_model_v1.h5", "keras_h5"),
        "lstm_h5": ("lstm_model_v1.h5", "keras_h5"),
        "sentiment_model": ("sentiment_model.joblib", "joblib"),
        "sentiment_vectorizer": ("sentiment_vectorizer.joblib", "joblib"),
    }
    for key, (fname, ftype) in map_.items():
        path = download_file_from_drive(key, fname)
        if path:
            try:
                if ftype == "joblib":
                    m = joblib.load(path)
                elif ftype == "keras_h5" and tf is not None:
                    m = tf.keras.models.load_model(path)
                else:
                    m = None
                loaded[key] = m
                msgs.append((key, True, f"loaded from {path}"))
                continue
            except Exception as e:
                msgs.append((key, False, f"downloaded but load failed: {e}"))
        else:
            msgs.append((key, False, "not found on Drive"))

        # fallback
        try:
            if key == "risk":
                loaded[key] = build_fallback_risk_model()
            elif key == "los":
                loaded[key] = build_fallback_los_model()
            elif key == "cnn_h5":
                loaded[key] = build_small_cnn()
            elif key == "lstm_h5":
                loaded[key] = build_small_lstm()
            elif key == "sentiment_model":
                # we'll keep a single pipeline under 'sentiment_model' (vectorizer+clf)
                loaded["sentiment_model"] = build_fallback_sentiment()
            elif key == "sentiment_vectorizer":
                # not needed because pipeline contains vectorizer
                loaded["sentiment_vectorizer"] = None
            msgs.append((key, True, "fallback created"))
        except Exception as e:
            msgs.append((key, False, f"fallback build failed: {e}"))
            loaded[key] = None
    return loaded, msgs

models, load_messages = load_models_with_fallback()

# show short notifications and vanish
for key, ok, info in load_messages:
    if ok:
        st.success(f"{key} ready — {info}")
    else:
        st.warning(f"{key} missing — {info}")

st.markdown("""<script>
setTimeout(()=>{document.querySelectorAll('.stAlert').forEach(e=>e.remove());},2500);
</script>""", unsafe_allow_html=True)

# -------------------------
# Sample data creators (so pages do not break)
# -------------------------
def ensure_sample_files():
    os.makedirs("data", exist_ok=True)
    # vitals_timeseries.csv (for LSTM page)
    vit = os.path.join("data","vitals_timeseries.csv")
    if not os.path.exists(vit):
        df = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=30, freq="H"),
            "heart_rate": (60 + np.sin(np.linspace(0,6,30))*5 + np.random.randn(30)*1).round(),
            "spo2": (97 + np.random.randn(30)*0.5).round(1),
            "resp_rate": (14 + np.random.randn(30)*0.8).round()
        })
        df.to_csv(vit, index=False)
    # tabular_complete.csv (for clustering)
    tab = os.path.join("data","tabular_complete.csv")
    if not os.path.exists(tab):
        df = pd.DataFrame(np.random.normal(size=(200,6)), columns=[f"f{i}" for i in range(6)])
        df.to_csv(tab, index=False)
    # transactions.csv (association)
    tx = os.path.join("data","transactions.csv")
    if not os.path.exists(tx):
        sample = ["apple,banana", "banana,carrot", "apple,carrot", "banana,donut", "apple,donut"]
        pd.DataFrame({"Items": sample}).to_csv(tx, index=False)

ensure_sample_files()

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose a module", [
    "Home",
    "Classification (Risk)",
    "Regression (LOS)",
    "Clustering",
    "Association Rules",
    "CNN Imaging",
    "LSTM Time Series",
    "Chatbot (Gemini)",
    "Translator",
    "Sentiment"
])

# -------------------------
# Pages
# -------------------------
if module == "Home":
    st.header("Welcome to HealthAI")
    st.write("This demo loads real models from Drive (if you supplied links in Streamlit secrets). If a model isn't present the app builds a tiny fallback so everything runs reliably for demos.")
    st.write("Make sure to set `GEMINI_API_KEY` (if you want chatbot) and Drive links under `[DRIVE]` in Streamlit secrets.")

# Classification Risk
elif module == "Classification (Risk)":
    st.header("Disease Risk Prediction")
    age = st.number_input("Age", 0, 120, 45)
    bp = st.number_input("Blood Pressure", 40, 220, 120)
    glucose = st.number_input("Glucose", 40, 400, 110)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    if st.button("Predict Risk"):
        model = models.get("risk")
        if model is None:
            st.error("Risk model unavailable")
        else:
            X = np.array([[age, bp, glucose, bmi]])
            try:
                pred = model.predict(X)
                proba = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
                label = "High risk" if int(pred[0])==1 else "Low risk"
                st.success(label)
                if proba is not None:
                    st.write(f"Probability score (positive): {proba[0]:.3f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Regression LOS
elif module == "Regression (LOS)":
    st.header("Length of Stay (LOS) prediction")
    age = st.number_input("Age", 0, 120, 50)
    severity = st.slider("Severity (1-5)", 1, 5, 2)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    if st.button("Predict LOS"):
        model = models.get("los")
        if model is None:
            st.error("LOS model missing")
        else:
            X = np.array([[age, severity, bmi]])
            try:
                pred = model.predict(X)
                st.success(f"Expected LOS: {float(pred[0]):.2f} days")
            except Exception as e:
                st.error(f"LOS prediction error: {e}")

# Clustering
elif module == "Clustering":
    st.header("Patient Clustering (KMeans + PCA)")
    uploaded = st.file_uploader("CSV with numeric features (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("data/tabular_complete.csv")
        st.info("Using sample data: data/tabular_complete.csv")
    numeric = df.select_dtypes(include=np.number)
    k = st.slider("K clusters", 2, 8, 3)
    if st.button("Run clustering"):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(numeric.fillna(0))
        labels = kmeans.labels_
        pca = PCA(n_components=2)
        red = pca.fit_transform(numeric.fillna(0))
        fig, ax = plt.subplots(figsize=(8,4))
        scatter = ax.scatter(red[:,0], red[:,1], c=labels, cmap="tab10")
        ax.set_title("PCA projection with cluster coloring")
        st.pyplot(fig)
        st.dataframe(pd.Series(labels).value_counts().rename("count"))

# Association Rules
elif module == "Association Rules":
    st.header("Association rule mining (Apriori)")
    csv_file = st.file_uploader("transactions CSV (col 'Items' comma-separated)", type=["csv"])
    if csv_file:
        tx_df = pd.read_csv(csv_file)
    else:
        tx_df = pd.read_csv("data/transactions.csv")
        st.info("Using sample: data/transactions.csv")
    if "Items" not in tx_df.columns:
        st.error("CSV must have column 'Items'")
    else:
        tx_list = tx_df["Items"].apply(lambda s: [x.strip() for x in str(s).split(",")])
        te = TransactionEncoder()
        te_ary = te.fit(tx_list).transform(tx_list)
        tdf = pd.DataFrame(te_ary, columns=te.columns_)
        freq = apriori(tdf, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.write("Top rules by lift")
        st.dataframe(rules.sort_values("lift", ascending=False).head(20))

# CNN Imaging
elif module == "CNN Imaging":
    st.header("CNN Imaging (example)")
    img_file = st.file_uploader("Upload chest xray image", type=["png","jpg","jpeg"])
    model = models.get("cnn_h5")
    if model is None:
        st.error("CNN model not available")
    else:
        st.info("Model available. Upload an image to run inference.")
        if img_file:
            img = Image.open(img_file).convert("RGB").resize((224,224))
            arr = np.array(img)/255.0
            x = np.expand_dims(arr,0).astype(np.float32)
            try:
                preds = model.predict(x)
                # interpret predict shape
                if preds.shape[-1] == 1:
                    score = float(preds[0])
                    label = "Pneumonia" if score>0.5 else "Normal"
                    st.write(f"{label} ({score:.3f})")
                else:
                    idx = int(np.argmax(preds[0]))
                    st.write(f"Class index {idx}, scores: {preds[0].tolist()}")
                # Try Grad-CAM if tf available
                if tf is not None:
                    try:
                        # find last conv
                        last_conv = None
                        for layer in reversed(model.layers):
                            if 'conv' in layer.name.lower():
                                last_conv = layer.name; break
                        if last_conv:
                            # Grad-CAM
                            heatmap = None
                            with tf.GradientTape() as tape:
                                grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
                                conv_outs, preds2 = grad_model(x)
                                class_idx = tf.argmax(preds2[0])
                                loss = preds2[:, class_idx]
                                grads = tape.gradient(loss, conv_outs)
                                pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                                conv_outs = conv_outs[0]
                                heatmap = conv_outs @ pooled_grads[..., tf.newaxis]
                                heatmap = tf.squeeze(heatmap).numpy()
                                heatmap = np.maximum(heatmap, 0); heatmap = heatmap/ (heatmap.max()+1e-8)
                                heatmap = (heatmap*255).astype("uint8")
                            # overlay
                            heat_img = Image.fromarray(heatmap).resize(img.size)
                            cmap = plt.cm.jet(heatmap/255.0)[:,:,:3]
                            cmap_img = Image.fromarray((cmap*255).astype("uint8"))
                            overlay = Image.blend(img.convert("RGBA"), cmap_img.convert("RGBA"), alpha=0.4)
                            st.image([img, overlay], caption=["Original","Grad-CAM Overlay"], width=300)
                        else:
                            st.warning("No conv layer found for Grad-CAM.")
                    except Exception as e:
                        st.warning(f"Grad-CAM error: {e}")
            except Exception as e:
                st.error(f"Inference failed: {e}")

# LSTM Time Series
elif module == "LSTM Time Series":
    st.header("LSTM time-series forecast (vitals)")
    uploaded = st.file_uploader("Upload vitals CSV (time + numeric columns)", type=["csv"])
    if uploaded:
        vit = pd.read_csv(uploaded)
    else:
        vit = pd.read_csv("data/vitals_timeseries.csv")
        st.info("Using sample: data/vitals_timeseries.csv")
    st.dataframe(vit.head())
    model = models.get("lstm_h5")
    if model is None:
        st.error("LSTM not available")
    else:
        # prepare numeric sequence (use first numeric column)
        num_cols = vit.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.error("No numeric columns in CSV")
        else:
            seq_col = num_cols[0]
            seq = vit[seq_col].values.astype(float)
            steps_required = None
            try:
                steps_required = model.input_shape[1]
            except Exception:
                steps_required = min(10, len(seq))
            st.info(f"Model expects {steps_required} timesteps (inferred).")
            if len(seq) < steps_required:
                st.warning("Not enough steps in sample for prediction.")
            else:
                last_seq = seq[-steps_required:]
                x = np.expand_dims(last_seq, 0).astype(np.float32)
                try:
                    pred = model.predict(x)
                    st.write(f"Predicted next {seq_col}: {float(pred[0][0]):.3f}")
                    # plot history + forecast
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(seq, label="history")
                    ax.scatter(len(seq), float(pred[0][0]), color="red", label="pred_next")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"LSTM predict failed: {e}")

# Chatbot (Gemini)
elif module == "Chatbot (Gemini)":
    st.header("Gemini Chatbot (short + explain)")
    q = st.text_input("Ask a medical question")
    short_btn = st.button("Get short answer")
    explain_btn = st.button("Explain in detail")
    if (short_btn or explain_btn) and q.strip():
        gem_key = st.secrets.get("GEMINI_API_KEY", "")
        if not gem_key:
            st.error("GEMINI_API_KEY missing in secrets.")
        elif genai is None:
            st.error("google.generativeai SDK not installed in environment.")
        else:
            try:
                genai.configure(api_key=gem_key)
                if short_btn:
                    prompt = f"You are a concise healthcare assistant. Answer in 1-2 sentences: {q}"
                else:
                    prompt = f"You are a detailed healthcare assistant. Provide an explanatory answer with headings: {q}"
                resp = genai.generate_content(prompt)
                text = getattr(resp, "text", None) or str(resp)
                st.write(text)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")

# Translator
elif module == "Translator":
    st.header("Translator (googletrans)")
    txt = st.text_area("Text to translate")
    dest = st.selectbox("Target language", ["en","ta","hi","ml"], index=0)
    if st.button("Translate"):
        if GT_Translator is None:
            st.error("googletrans not installed")
        else:
            try:
                tr = GT_Translator()
                out = tr.translate(txt, dest=dest)
                st.write(out.text)
            except Exception as e:
                st.error(f"Translate failed: {e}")

# Sentiment
elif module == "Sentiment":
    st.header("Sentiment Analysis")
    txt = st.text_area("Enter feedback text")
    if st.button("Analyze"):
        clf = models.get("sentiment_model")
        if clf is None:
            st.error("Sentiment model missing")
        else:
            try:
                pred = clf.predict([txt])[0]
                label = "Positive" if int(pred)==1 else "Negative"
                st.success(f"Sentiment: {label}")
            except Exception as e:
                st.error(f"Sentiment prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Notes: Put Drive links under [DRIVE] in Streamlit secrets (keys: risk, los, cnn_h5, lstm_h5, sentiment_model, sentiment_vectorizer). Put GEMINI_API_KEY in secrets for chatbot.")
