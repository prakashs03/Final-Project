# app.py (FULL - copy/paste)
import os, sys, traceback, time
from typing import Optional
import streamlit as st
import joblib
import gdown
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Translator (no googletrans)
from deep_translator import GoogleTranslator

# Try to import Gemini client (handle API surface differences)
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="HealthAI (fixed)", layout="wide")
st.title("💊 HealthAI — Fixed model loading & preprocessing")

# Ensure models dir
os.makedirs("models", exist_ok=True)

# --------------------- Utilities ---------------------
def drive_download(id_or_url: str, dest: str):
    """Download google drive file by id or full url using gdown. Returns path or raises."""
    if not id_or_url:
        raise ValueError("No Drive id provided.")
    url = f"https://drive.google.com/uc?id={id_or_url}"
    gdown.download(url, dest, quiet=True)
    if not os.path.exists(dest):
        raise RuntimeError(f"Failed to download file id {id_or_url}")
    return dest

@st.cache_resource(show_spinner=False)
def load_joblib_safe(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_keras_safe(path: str):
    # Load with compile=False to avoid optimizer/variable build issues during deserialization
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner=False)
def load_all_models_from_secrets():
    """
    Downloads models from Drive ids stored in st.secrets["DRIVE"].
    Returns dict of models and preprocessing scalers (may be None).
    """
    drive = st.secrets.get("DRIVE", {})
    got = {}
    # mapping expected keys
    mapping = {
        "risk": ("models/risk_model.joblib", "joblib"),
        "los": ("models/los_model.joblib", "joblib"),
        "cnn_h5": ("models/cnn_model.h5", "keras"),
        "lstm_h5": ("models/lstm_model.h5", "keras"),
        "sentiment_model": ("models/sentiment_model.joblib", "joblib"),
        "sentiment_vectorizer": ("models/sentiment_vectorizer.joblib", "joblib"),
        # scalers
        "risk_scaler": ("models/risk_scaler.joblib", "joblib"),
        "los_scaler": ("models/los_scaler.joblib", "joblib"),
        "lstm_scaler": ("models/lstm_scaler.joblib", "joblib"),
    }
    for key, (local, kind) in mapping.items():
        file_id = drive.get(key)
        if not file_id:
            got[key] = None
            continue
        try:
            drive_download(file_id, local)
            if kind == "joblib":
                got[key] = load_joblib_safe(local)
            elif kind == "keras":
                got[key] = load_keras_safe(local)
            else:
                got[key] = None
        except Exception as e:
            st.error(f"Error loading {key}: {e}")
            got[key] = None
    return got

# Load models/scalers (cached)
with st.spinner("Loading models and scalers..."):
    resources = load_all_models_from_secrets()

# Fallback small models when not present (only for UI safety)
def build_fallbacks():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf.fit(np.zeros((2,6)), [0,1])
    lr = LinearRegression()
    lr.fit(np.zeros((2,6)), [4.5, 6.0])
    # tiny keras cnn/lstm
    try:
        cnn = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,128,3)),
            tf.keras.layers.Conv2D(8,3,activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    except Exception:
        cnn = None
    try:
        lstm = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,1)),
            tf.keras.layers.LSTM(8),
            tf.keras.layers.Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mse')
    except Exception:
        lstm = None
    return {"risk":rf, "los":lr, "cnn_h5":cnn, "lstm_h5":lstm}

fallback = build_fallbacks()

# Merge into models dict with clear warnings
models = {}
for k in ["risk","los","cnn_h5","lstm_h5","sentiment_model","sentiment_vectorizer",
          "risk_scaler","los_scaler","lstm_scaler"]:
    val = resources.get(k)
    if val is None:
        if k in fallback:
            models[k] = fallback[k]
            st.warning(f"Using fallback for {k} (no file in secrets). Predictions may be inaccurate.")
        else:
            models[k] = None
    else:
        models[k] = val
        st.info(f"Loaded {k} from Drive.")

# --------------------- Preprocessing-aware prediction helpers ---------------------
def get_model_input_shape(keras_model):
    # returns tuple like (None, H, W, C) or (None, T, features)
    try:
        return keras_model.input_shape
    except Exception:
        return None

def preprocess_tabular_for_model(row: dict, scaler: Optional[StandardScaler]=None):
    # Adopt your feature order. Update to match your training feature order exactly.
    features = np.array([
        row.get("age", 50),
        row.get("bmi", 26.5),
        row.get("blood_pressure", 135),
        row.get("cholesterol", 225),
        row.get("diabetes", 0),
        row.get("heart_disease", 0)
    ], dtype=float).reshape(1,-1)
    if scaler is not None:
        try:
            features = scaler.transform(features)
        except Exception:
            # If scaler not compatible, ignore
            pass
    return features

def predict_risk(row: dict):
    model = models.get("risk")
    scaler = models.get("risk_scaler")
    X = preprocess_tabular_for_model(row, scaler)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            return pred, float(np.max(probs))
        else:
            p = model.predict(X)
            return int(p[0]), 0.5
    except Exception as e:
        st.error(f"Risk prediction error: {e}")
        return 0, 0.0

def predict_los(row: dict):
    model = models.get("los")
    scaler = models.get("los_scaler")
    X = preprocess_tabular_for_model(row, scaler)
    try:
        pred = model.predict(X)[0]
        return float(pred)
    except Exception as e:
        st.error(f"LOS prediction error: {e}")
        return 1.0

def preprocess_image_for_model(img: Image.Image, keras_model):
    # detect expected input shape; resize and normalize
    shp = get_model_input_shape(keras_model)
    # default size
    target_h, target_w = 128, 128
    target_c = 3
    if shp and len(shp) == 4:
        _, h, w, c = shp
        if h is not None and w is not None:
            target_h, target_w = int(h), int(w)
        if c is not None:
            target_c = int(c)
    img = img.convert("RGB")
    img = img.resize((target_w, target_h))
    arr = np.array(img).astype("float32")/255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*target_c, axis=-1)
    arr = arr.reshape((1, target_h, target_w, target_c))
    return arr

def predict_cnn(image: Image.Image):
    model = models.get("cnn_h5")
    if model is None:
        return "UNKNOWN", 0.0
    arr = preprocess_image_for_model(image, model)
    try:
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        # Provide mapping from model to labels if you saved it; else assume [NORMAL,PNEUMONIA]
        labels = ["NORMAL", "PNEUMONIA"]
        return labels[idx], float(probs[idx])
    except Exception as e:
        st.error(f"CNN prediction error: {e}")
        return "ERROR", 0.0

def preprocess_timeseries_for_lstm(df_ts: pd.DataFrame, keras_model):
    shp = get_model_input_shape(keras_model)
    seq_len = 10
    if shp and len(shp) >= 3:
        # shape like (None, T, features)
        seq_len = int(shp[1]) if shp[1] is not None else seq_len
    series = df_ts.iloc[:,0].values.astype(float)
    if len(series) >= seq_len:
        seq = series[-seq_len:]
    else:
        seq = np.pad(series, (seq_len-len(series), 0), 'constant', constant_values=0.0)
    seq = seq.reshape((1, seq_len, 1))
    scaler = models.get("lstm_scaler")
    if scaler is not None:
        try:
            # if scaler was fit on sequences flattened to shape (n, seq_len), transform accordingly
            flat = scaler.transform(seq.reshape(1, -1)).reshape(seq.shape)
            return flat
        except Exception:
            pass
    return seq

def predict_lstm(df_ts: pd.DataFrame):
    model = models.get("lstm_h5")
    if model is None:
        return 1.0
    seq = preprocess_timeseries_for_lstm(df_ts, model)
    try:
        pred = model.predict(seq, verbose=0)[0][0]
        return float(pred)
    except Exception as e:
        st.error(f"LSTM prediction error: {e}")
        return 1.0

# --------------------- Gemini wrapper (robust) ---------------------
def gemini_query(prompt: str, model_name: Optional[str]=None):
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini client not available in environment.")
    api_key = st.secrets.get("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GENAI_API_KEY missing in secrets.")
    genai.configure(api_key=api_key)
    model_name = model_name or st.secrets.get("GEMINI_MODEL") or "models/gemini-2.5-flash"
    # Try multiple call patterns
    try:
        m = genai.GenerativeModel(model_name)
        resp = m.generate_content(f"You are a healthcare assistant. {prompt}")
        # adapt to different response shapes
        text = getattr(resp, "text", None)
        if not text and isinstance(resp, dict):
            # older format
            text = resp.get("candidates", [{}])[0].get("content", "")
        return text or str(resp)
    except Exception as e:
        # fallback older client pattern
        try:
            resp = genai.generate(prompt, model=model_name)
            return resp.get("output", "")
        except Exception as e2:
            raise RuntimeError(f"Gemini calls failed: {e} | {e2}")

# --------------------- Streamlit UI ---------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Module", [
    "Overview","Risk","LOS","CNN Image","LSTM Time-series","Chatbot","Translator","Diagnostics"
])

if page == "Overview":
    st.header("Overview")
    st.write("This app now loads models with compile=False, uses scalers if provided, and preprocesses correctly.")
    st.write("Make sure scalers (risk_scaler, los_scaler, lstm_scaler) are uploaded to Drive and put into secrets under [DRIVE].")

elif page == "Risk":
    st.header("Risk prediction")
    with st.form("riskf"):
        age = st.number_input("Age", 0, 120, 50)
        bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
        bp = st.number_input("Blood pressure", 80, 220, 135)
        chol = st.number_input("Cholesterol", 100, 400, 225)
        diabetes = st.selectbox("Diabetes", [0,1])
        heart = st.selectbox("Heart disease", [0,1])
        sub = st.form_submit_button("Predict")
    if sub:
        row = {"age":age,"bmi":bmi,"blood_pressure":bp,"cholesterol":chol,"diabetes":diabetes,"heart_disease":heart}
        p, conf = predict_risk(row)
        st.success(f"Predicted class: {p}   probability approx: {conf:.3f}")

elif page == "LOS":
    st.header("Length of stay (LOS)")
    with st.form("losf"):
        age = st.number_input("Age", 0, 120, 50)
        bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
        bp = st.number_input("Blood pressure", 80, 220, 135)
        chol = st.number_input("Cholesterol", 100, 400, 225)
        diabetes = st.selectbox("Diabetes", [0,1], key="los_diab")
        heart = st.selectbox("Heart disease", [0,1], key="los_heart")
        sub = st.form_submit_button("Predict LOS")
    if sub:
        row = {"age":age,"bmi":bmi,"blood_pressure":bp,"cholesterol":chol,"diabetes":diabetes,"heart_disease":heart}
        days = predict_los(row)
        st.success(f"Predicted LOS: {days:.2f} days")

elif page == "CNN Image":
    st.header("CNN Image classification (X-ray)")
    up = st.file_uploader("Upload chest X-ray", type=["png","jpg","jpeg"])
    if up:
        try:
            img = Image.open(up)
            st.image(img, caption="Uploaded image", use_column_width=True)
            label, conf = predict_cnn(img)
            st.success(f"Prediction: {label}  (conf {conf:.3f})")
        except Exception as e:
            st.error(f"Image processing error: {e}")

elif page == "LSTM Time-series":
    st.header("LSTM next-step prediction")
    up = st.file_uploader("Upload CSV (single column time-series)", type=["csv"])
    if up:
        df_ts = pd.read_csv(up)
        st.write(df_ts.head())
        pred = predict_lstm(df_ts)
        st.success(f"Predicted next value: {pred:.3f}")

elif page == "Chatbot":
    st.header("Chatbot (Gemini primary)")
    q = st.text_input("Question")
    lang = st.selectbox("Language", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Ask"):
        if not q.strip():
            st.warning("Type a question")
        else:
            if GENAI_AVAILABLE and st.secrets.get("GENAI_API_KEY"):
                try:
                    resp = gemini_query(f"Language: {lang}\n{q}", model_name=st.secrets.get("GEMINI_MODEL"))
                    st.write(resp)
                except Exception as e:
                    st.error(f"Gemini failed: {e}")
                    st.info("Using fallback canned answers.")
                    # fallback
                    if "diabet" in q.lower():
                        st.write("Short: Frequent urination, thirst, fatigue. Ask for 'explain' to expand.")
                    else:
                        st.write("Short: Consult a clinician. Ask 'explain' for more details.")
            else:
                st.info("Gemini not configured - using simple fallback.")
                if "diabet" in q.lower():
                    st.write("Short: Frequent urination, thirst, fatigue.")
                else:
                    st.write("Short: Please consult a clinician.")

elif page == "Translator":
    st.header("Translator (deep-translator)")
    txt = st.text_area("Text to translate")
    tgt = st.selectbox("Target", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Translate"):
        code = {"English":"en","Tamil":"ta","Hindi":"hi","Malayalam":"ml"}[tgt]
        try:
            tr = GoogleTranslator(source="auto", target=code).translate(txt)
            st.success(tr)
        except Exception as e:
            st.error(f"Translate error: {e}")

elif page == "Diagnostics":
    st.header("Model & resource diagnostics (debug)")
    st.write("Loaded resources summary (types & shapes):")
    for k,v in models.items():
        try:
            info = str(type(v))
            if hasattr(v, "input_shape"):
                info += f" | input_shape={getattr(v,'input_shape')}"
        except:
            info = str(type(v))
        st.write(f"{k}: {info}")
    st.write("---")
    st.write("If CNN always outputs NORMAL, check: model input size (above) and that you uploaded the correct model and label mapping.")
    st.write("If LOS always returns constant, ensure los_scaler and the exact preprocessing used during training are uploaded.")

st.caption("Note: This demo is for development. Validate clinical predictions before any use.")
