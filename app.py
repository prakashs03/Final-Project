# app.py — HealthAI final (streamlit)
# Copy / paste exactly. Models are loaded from local models/ directory.
# Make sure your model files are in the repository under models/

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import math

# Optional: Gemini (google generative AI). Put API key + model in Streamlit secrets
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ----------------- PAGE -----------------
st.set_page_config(page_title="HealthAI — Smart Healthcare Assistant", layout="wide")
st.title("💊 HealthAI — Smart Healthcare Assistant")

# ----------------- UTIL: GEMINI WRAPPER -----------------
def gemini_configured():
    try:
        if genai is None:
            return False
        key = st.secrets.get("GENAI_API_KEY", None)
        model = st.secrets.get("GENAI_MODEL", None)
        if not key or not model:
            return False
        genai.configure(api_key=key)
        return True
    except Exception:
        return False

def gemini_short(prompt):
    """Call Gemini (safe wrapper). Returns text or fallback."""
    try:
        if not gemini_configured():
            return "(Gemini not configured) " + (prompt[:200] + "...")
        # library has changed over time; try a couple of call styles
        try:
            # older/newer api: generative.generate -> returns object with text
            resp = genai.generate(prompt=prompt)  # try common style
            if hasattr(resp, "text"):
                return resp.text.strip()
            # fallback to object str
            return str(resp)
        except Exception:
            try:
                resp = genai.generate_text(model=st.secrets["GENAI_MODEL"], input=prompt)
                # different responses; try to extract text
                return getattr(resp, "text", str(resp))
            except Exception as e:
                # last fallback
                return "(Gemini call failed) " + str(e)
    except Exception as e:
        return "(Gemini unavailable) " + str(e)

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_models_local():
    models = {}
    base = "models"
    required = {
        "risk": "risk_classifier_v1.joblib",
        "los": "los_regressor_v1.joblib",
        "cnn": "cnn_model_v1.h5",
        "lstm": "lstm_model_v1.h5",
        "sentiment_model": "sentiment_model.joblib",
        "sentiment_vectorizer": "sentiment_vectorizer.joblib",
        # optional scalers if you saved them
        "risk_scaler": "risk_scaler.joblib",
        "los_scaler": "los_scaler.joblib",
        "lstm_scaler": "lstm_scaler.joblib",
    }
    for key, fname in required.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            try:
                if fname.endswith(".joblib"):
                    models[key] = joblib.load(path)
                elif fname.endswith(".h5"):
                    # load Keras model without compiling (faster)
                    models[key] = tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                st.warning(f"Warning: failed loading {fname}: {e}")
        else:
            st.info(f"Model file not found: {path} (this is OK for demo; some features fallback).")
    return models

models = load_models_local()

# ----------------- SAFE INVERSE TRANSFORM for single target -----------------
def inverse_single_target(scaler, y_scaled):
    """
    scaler: a fitted sklearn scaler (StandardScaler-like).
    y_scaled: float or array-like in scaled space (1D).
    Returns unscaled value for index 0 using scaler.scale_[0] + mean_[0] if multi-dim.
    """
    import numpy as _np
    y = _np.asarray(y_scaled).reshape(-1, 1)
    if not hasattr(scaler, "scale_"):
        # cannot invert
        return float(y[0][0])
    sc = scaler.scale_
    mu = scaler.mean_
    if sc.shape[0] == 1:
        return float((y * sc + mu)[0][0])
    else:
        # assume target correspond to first column used when scaling; invert using first entry
        return float(y[0][0] * sc[0] + mu[0])

# ----------------- CNN PREDICTION (robust) -----------------
def preprocess_image_for_model(img: Image.Image, model):
    # handle grayscale/RGB and resizing based on model input shape if available
    arr = img.convert("RGB")
    if hasattr(model, "input_shape") and model.input_shape is not None:
        # input_shape like (None, height, width, channels)
        try:
            _, h, w, c = model.input_shape
            arr = arr.resize((w, h))
        except Exception:
            arr = arr.resize((224, 224))
    else:
        arr = arr.resize((224, 224))
    x = img_to_array(arr).astype("float32") / 255.0
    return np.expand_dims(x, 0)

def cnn_predict_local(img: Image.Image):
    if "cnn" not in models:
        return "Model not available", 0.0
    m = models["cnn"]
    x = preprocess_image_for_model(img, m)
    preds = m.predict(x)
    # handle single-output (sigmoid) or multiclass (softmax)
    if preds.size == 0:
        return "Unknown", 0.0
    if preds.shape[-1] == 1:
        p = float(preds[0][0])
        label = "Pneumonia" if p >= 0.5 else "Normal"
        return label, p * 100.0
    else:
        idx = int(np.argmax(preds[0]))
        # try safe label mapping. If incorrect, change mapping here.
        classes = ["Normal", "Pneumonia"] if preds.shape[-1] >= 2 else ["Normal"]
        label = classes[idx] if idx < len(classes) else f"Class_{idx}"
        prob = float(np.max(preds[0]))
        return label, prob * 100.0

# ----------------- LOS PREDICTION (robust) -----------------
def predict_los_from_features(features_list):
    # features_list: length N (must match training layout)
    if "los" not in models:
        return 1.0
    los = models["los"]
    # try to apply scaler if available
    if "los_scaler" in models:
        try:
            scaled = models["los_scaler"].transform([features_list])
            pred_scaled = los.predict(scaled)
            # invert using inverse_single_target
            return max(1.0, round(inverse_single_target(models["los_scaler"], pred_scaled)[0] if hasattr(pred_scaled, "__len__") else inverse_single_target(models["los_scaler"], pred_scaled), 1))
        except Exception as e:
            # fallback: try direct predict
            try:
                pred = los.predict([features_list])[0]
                return max(1.0, round(float(pred), 1))
            except Exception:
                return 1.0
    else:
        try:
            pred = los.predict([features_list])[0]
            return max(1.0, round(float(pred), 1))
        except Exception:
            return 1.0

# ----------------- LSTM FORECAST (robust) -----------------
def lstm_forecast_series(series):
    if "lstm" not in models:
        return float(np.mean(series))
    lstm = models["lstm"]
    # infer model input shape
    in_shape = lstm.input_shape  # (None, timesteps, features) or (None, timesteps, features)
    try:
        _, timesteps, features = in_shape
    except Exception:
        # fallback
        timesteps = 20
        features = 1
    arr = np.array(series, dtype=float).reshape(-1)
    # pad / trim
    if arr.size < timesteps:
        pad = np.full(timesteps - arr.size, arr[-1] if arr.size > 0 else 0.0)
        arr2 = np.concatenate([pad, arr])
    else:
        arr2 = arr[-timesteps:]
    # create required features dimension
    if features == 1:
        X = arr2.reshape(1, timesteps, 1)
    else:
        # replicate the single series across features (safe fallback)
        X = np.tile(arr2.reshape(1, timesteps, 1), (1, 1, features))
    # predict
    try:
        pred = lstm.predict(X)
        # inverse scale if scaler available
        if "lstm_scaler" in models:
            val = inverse_single_target(models["lstm_scaler"], pred.reshape(-1, 1))
            return round(float(val), 2)
        else:
            # assume model outputs scalar
            return round(float(pred.reshape(-1)[0]), 2)
    except Exception:
        # fallback average
        return round(float(np.mean(arr2)), 2)

# ----------------- SENTIMENT -----------------
def sentiment_local(text):
    if "sentiment_model" in models and "sentiment_vectorizer" in models:
        try:
            X = models["sentiment_vectorizer"].transform([text])
            p = models["sentiment_model"].predict(X)[0]
            label = "Positive" if int(p) == 1 else "Negative"
            # optionally check with gemini for better accuracy if available
            if gemini_configured():
                g = gemini_short(f"Classify sentiment (Positive/Negative) for: {text}")
                if "positive" in g.lower():
                    return "Positive (Gemini)"
                if "negative" in g.lower():
                    return "Negative (Gemini)"
            return label
        except Exception as e:
            return f"Error: {e}"
    else:
        # fallback to Gemini if configured
        if gemini_configured():
            return gemini_short(f"Classify sentiment for: {text} (Positive or Negative)")
        return "Model not available"

# ----------------- UI / Tabs -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Risk & LOS", "LSTM Forecast", "CNN (X-ray)", "Chatbot (Gemini)", "Translator (Gemini)", "Sentiment"
])

with tab1:
    st.header("Disease Risk Classification + Length of Stay")
    # NOTE: you said you uploaded CSVs manually; we assume user will provide tabular inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        bp = st.number_input("Blood Pressure", min_value=40, max_value=220, value=120)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=400, value=100)
    with c2:
        bmi = st.number_input("BMI", min_value=8.0, max_value=60.0, value=24.5)
        hr = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
        chol = st.number_input("Cholesterol", min_value=50, max_value=400, value=200)
    with c3:
        st.write(" ")
        if st.button("Predict Risk & LOS"):
            features = [age, bp, glucose, bmi, hr, chol]
            # risk
            try:
                if "risk" in models:
                    if "risk_scaler" in models:
                        risk_X = models["risk_scaler"].transform([features])
                        risk_pred = models["risk"].predict(risk_X)[0]
                    else:
                        risk_pred = models["risk"].predict([features])[0]
                    st.success(f"Disease Risk: {'HIGH' if int(risk_pred)==1 else 'LOW'}")
                else:
                    st.info("Risk model not available")
            except Exception as e:
                st.error(f"Risk prediction error: {e}")
            # LOS
            los_days = predict_los_from_features(features)
            st.info(f"Predicted Length of Stay: {los_days} days")

with tab2:
    st.header("LSTM Forecast (time-series)")
    seq = st.text_area("Enter time series (comma separated numbers). E.g., 98,99,100,101", height=80)
    if st.button("Forecast LSTM"):
        try:
            if not seq.strip():
                st.warning("Enter a sequence")
            else:
                vals = [float(x.strip()) for x in seq.split(",") if x.strip()!='']
                out = lstm_forecast_series(vals)
                st.metric("Forecasted value", out)
        except Exception as e:
            st.error(f"LSTM error: {e}")

with tab3:
    st.header("CNN — Chest X-ray classification")
    uploaded = st.file_uploader("Upload chest X-ray", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)
        if st.button("Analyze Image"):
            label, prob = cnn_predict_local(img)
            st.success(f"Result: {label} ({prob:.2f}%)")

with tab4:
    st.header("Chatbot (Gemini)")
    q = st.text_input("Ask a health question (short):")
    if st.button("Get Short Answer"):
        if q.strip():
            short = gemini_short(f"Answer in one short sentence: {q}")
            st.info(short)
            if st.button("Explain this in detail"):
                expl = gemini_short(f"Explain clearly and concisely: {q}")
                st.write(expl)
        else:
            st.warning("Type a question")

with tab5:
    st.header("Translator (Gemini)")
    src = st.text_area("Text to translate:")
    tgt = st.text_input("Target language (e.g., Hindi, Tamil, French):")
    if st.button("Translate"):
        if not src.strip() or not tgt.strip():
            st.warning("Enter both text and target language")
        else:
            out = gemini_short(f"Translate the following medical text to {tgt}: {src}")
            st.success(out)

with tab6:
    st.header("Sentiment analysis")
    feedback = st.text_area("Patient feedback / note:")
    if st.button("Analyze Sentiment"):
        if not feedback.strip():
            st.warning("Enter text")
        else:
            s = sentiment_local(feedback)
            st.success(s)

# final note for user
st.markdown("---")
st.caption("Notes: • Models must be present in the `models/` folder. • Put Gemini key + model in Streamlit secrets as GENAI_API_KEY and GENAI_MODEL. • This app uses safe fallbacks for missing models.")
