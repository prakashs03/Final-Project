# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os
import traceback

# -- tensorflow may be heavy, import lazily only when needed --
def lazy_tf_import():
    import tensorflow as tf
    # patch InputLayer compatibility (some saved models include batch_shape)
    try:
        from tensorflow.keras.saving import register_keras_serializable
        from tensorflow.keras.layers import InputLayer

        @register_keras_serializable()
        class CompatibleInputLayer(InputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)
                super().__init__(*args, **kwargs)
    except Exception:
        pass
    return tf

# Gemini (Google Generative AI) lazy wrapper
def configure_gemini():
    # If secrets not present, returns None
    try:
        import google.generativeai as genai
    except Exception:
        return None, None
    key = st.secrets.get("GENAI_API_KEY")
    model_name = st.secrets.get("GENAI_MODEL")
    if not key:
        return None, None
    genai.configure(api_key=key)
    return genai, model_name

genai, GENAI_MODEL = configure_gemini()

st.set_page_config(page_title="💊 HealthAI - Smart Healthcare Assistant", layout="wide")
st.title("💊 HealthAI - Smart Healthcare Assistant")

# Utility: safe load local model
@st.cache_resource
def load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.session_state.setdefault("_load_errors", []).append(f"{os.path.basename(path)}: {e}")
        return None

@st.cache_resource
def load_keras(path):
    try:
        tf = lazy_tf_import()
        # compile=False to avoid optimizer/compile issues
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.session_state.setdefault("_load_errors", []).append(f"{os.path.basename(path)}: {e}")
        return None

# Where models should be located (repo /models folder)
MODEL_DIR = "models"

# Helper to find models (user should upload these exact names into /models)
MODEL_FILES = dict(
    risk = os.path.join(MODEL_DIR, "risk_classifier_v1.joblib"),
    risk_scaler = os.path.join(MODEL_DIR, "risk_scaler.joblib"),
    los = os.path.join(MODEL_DIR, "los_regressor_v1.joblib"),
    los_scaler = os.path.join(MODEL_DIR, "los_scaler.joblib"),   # may be scaler for X or y
    cnn = os.path.join(MODEL_DIR, "cnn_model_v1.h5"),
    lstm = os.path.join(MODEL_DIR, "lstm_model_v1.h5"),
    lstm_scaler = os.path.join(MODEL_DIR, "lstm_scaler.joblib"),
    sentiment_model = os.path.join(MODEL_DIR, "sentiment_model.joblib"),
    sentiment_vectorizer = os.path.join(MODEL_DIR, "sentiment_vectorizer.joblib"),
    patient_cluster = os.path.join(MODEL_DIR, "patient_cluster_model.joblib"),
)

# Show model loading status area
with st.expander("Model status & diagnostics (click to open)"):
    load_now = st.button("Load all models now (lazy loads otherwise)")
    if load_now:
        st.info("Loading models — this may take a few seconds...")
    models = {}
    # attempt loads only when user asked or when used (lazy)
    if load_now:
        # joblib models
        for key in ["risk", "los", "los_scaler", "risk_scaler", "lstm_scaler",
                    "sentiment_model", "sentiment_vectorizer", "patient_cluster"]:
            path = MODEL_FILES.get(key)
            if path and os.path.exists(path):
                models[key] = load_joblib(path)
            else:
                st.warning(f"{key} not found at {path}")
        # keras models
        for key in ["cnn", "lstm"]:
            path = MODEL_FILES.get(key)
            if path and os.path.exists(path):
                models[key] = load_keras(path)
            else:
                st.warning(f"{key} not found at {path}")

    # show any load errors captured
    if "_load_errors" in st.session_state:
        st.error("Model load warnings/errors:")
        for e in st.session_state["_load_errors"]:
            st.write("-", e)
    else:
        st.success("No load errors recorded (models will load on demand).")

# ---- Sidebar: allow user to upload CSVs/images if desired ----
st.sidebar.header("Data / Models")
uploaded_csv = st.sidebar.file_uploader("Upload tabular CSV (for LSTM / tests)", type=["csv"])
uploaded_xray = st.sidebar.file_uploader("Upload X-ray (jpeg/png) to test CNN", type=["png", "jpg", "jpeg"])

# Tabs for modules
tabs = st.tabs(["Risk & LOS", "LSTM Forecast", "CNN (X-ray)", "Chatbot / Translator", "Sentiment", "Clustering/Association"])

# -----------------------
# Tab: Risk & LOS
# -----------------------
with tabs[0]:
    st.header("Risk classification & Length-of-Stay (LOS) prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 45)
        bp = st.number_input("Blood Pressure", 30, 250, 120)
        glucose = st.number_input("Glucose", 30, 400, 100)
    with col2:
        bmi = st.number_input("BMI", 8.0, 60.0, 24.5)
        hr = st.number_input("Heart Rate", 20, 220, 80)
        chol = st.number_input("Cholesterol", 80, 500, 200)

    if st.button("Predict risk & LOS"):
        features = np.array([[age, bp, glucose, bmi, hr, chol]], dtype=float)
        # load on demand
        risk_model = load_joblib(MODEL_FILES["risk"]) if os.path.exists(MODEL_FILES["risk"]) else None
        los_model = load_joblib(MODEL_FILES["los"]) if os.path.exists(MODEL_FILES["los"]) else None
        los_scaler = load_joblib(MODEL_FILES["los_scaler"]) if os.path.exists(MODEL_FILES["los_scaler"]) else None

        # Risk
        try:
            if risk_model is None:
                st.warning("Risk model not found. Please add risk_classifier_v1.joblib in models/")
            else:
                # try probability if available
                if hasattr(risk_model, "predict_proba"):
                    prob = risk_model.predict_proba(features)[0][1]
                    label = "High" if prob >= 0.5 else "Low"
                    st.metric("Disease risk", label, f"{prob:.2%}")
                else:
                    pred = risk_model.predict(features)[0]
                    st.metric("Disease risk", "High" if pred == 1 else "Low")
        except Exception as e:
            st.error("Risk prediction failed: " + str(e))
            st.text(traceback.format_exc())

        # LOS
        try:
            if los_model is None:
                st.warning("LOS model not found. Please add los_regressor_v1.joblib in models/")
            else:
                # If a scaler exists, try to transform inputs before predicting
                X_to_pred = features
                if los_scaler is not None:
                    try:
                        # If scaler expects same number of input features:
                        if hasattr(los_scaler, "n_features_in_") and los_scaler.n_features_in_ == features.shape[1]:
                            X_to_pred = los_scaler.transform(features)
                        else:
                            # scaler may be a y-scaler or something else — skip scaling
                            pass
                    except Exception:
                        pass

                raw_pred = los_model.predict(X_to_pred)
                # attempt to inverse_transform if los_scaler is a y-scaler with 1 feature
                los_days = None
                if los_scaler is not None:
                    try:
                        # if scaler appears to be fitted on 1-dimensional targets
                        if hasattr(los_scaler, "n_features_in_") and los_scaler.n_features_in_ == 1:
                            los_days = los_scaler.inverse_transform(np.array(raw_pred).reshape(-1, 1))[0, 0]
                        else:
                            # can't inverse transform — assume regressor returns days already
                            los_days = float(raw_pred[0])
                    except Exception:
                        # fallback: use raw prediction value
                        los_days = float(raw_pred[0])
                else:
                    los_days = float(raw_pred[0])

                st.metric("Predicted LOS (days)", f"{los_days:.2f}")
        except Exception as e:
            st.error("LOS prediction failed: " + str(e))
            st.text(traceback.format_exc())

# -----------------------
# Tab: LSTM Forecast
# -----------------------
with tabs[1]:
    st.header("LSTM Forecast (vitals / timeseries)")

    df_file = uploaded_csv or st.file_uploader("Upload vitals CSV (rows=time, cols=features) for LSTM", type=["csv"])
    if df_file:
        try:
            df = pd.read_csv(df_file)
            st.dataframe(df.head())
            lstm_model = load_keras(MODEL_FILES["lstm"]) if os.path.exists(MODEL_FILES["lstm"]) else None
            lstm_scaler = load_joblib(MODEL_FILES["lstm_scaler"]) if os.path.exists(MODEL_FILES["lstm_scaler"]) else None
            if lstm_model is None:
                st.warning("LSTM model missing (models/lstm_model_v1.h5).")
            else:
                # Preprocess: scale if scaler available; otherwise use raw
                arr = df.values.astype(float)
                if lstm_scaler is not None:
                    try:
                        arr = lstm_scaler.transform(arr)
                    except Exception:
                        st.warning("lstm_scaler couldn't transform uploaded CSV; using raw values.")
                # Determine expected model input shape: (None, timesteps, features)
                try:
                    input_shape = lstm_model.input_shape  # often (None, timesteps, features)
                except Exception:
                    input_shape = None
                if input_shape is not None and len(input_shape) >= 3:
                    _, expected_t, expected_f = input_shape[0]
                    # reshape or truncate/pad as necessary
                    if arr.ndim == 2:
                        # if user uploaded whole series, reshape to (1, timesteps, features) if matches
                        t, f = arr.shape
                        if f != expected_f and expected_f is not None:
                            if expected_f <= f:
                                arr = arr[:, :expected_f]
                            else:
                                # pad
                                pad_cols = expected_f - f
                                arr = np.pad(arr, ((0,0),(0,pad_cols)), mode="constant")
                        # Now make batch dimension
                        batched = np.expand_dims(arr, axis=0)
                        # if expected_t differs: if expected_t < t, take last expected_t timesteps
                        if expected_t is not None and expected_t != t:
                            if expected_t < t:
                                batched = batched[:, -expected_t:, :]
                            else:
                                # pad timesteps with zeros
                                pad_ts = expected_t - t
                                batched = np.pad(batched, ((0,0),(pad_ts,0),(0,0)), mode="constant")
                    else:
                        batched = arr
                else:
                    # fallback: try simple reshape (1, n, 1)
                    batched = arr.reshape(1, arr.shape[0], arr.shape[1] if arr.ndim>1 else 1)
                try:
                    pred = lstm_model.predict(batched)
                    # If pred is sequence output, take last step
                    if pred.ndim == 3:
                        out = pred[0, -1, :]
                    else:
                        out = pred[0]
                    # inverse transform if scaler exists and appropriate
                    if lstm_scaler is not None:
                        try:
                            inv = lstm_scaler.inverse_transform(out.reshape(1, -1))
                            st.write("Forecast (rescaled):")
                            st.line_chart(pd.DataFrame(inv))
                        except Exception:
                            st.write("Forecast (raw):")
                            st.line_chart(pd.DataFrame(out.reshape(1, -1)))
                    else:
                        st.write("Forecast (raw):")
                        st.line_chart(pd.DataFrame(out.reshape(1, -1)))
                except Exception as e:
                    st.error("LSTM prediction failed: " + str(e))
                    st.text(traceback.format_exc())
        except Exception as e:
            st.error("Failed to read uploaded CSV: " + str(e))

# -----------------------
# Tab: CNN (X-ray)
# -----------------------
with tabs[2]:
    st.header("CNN X-ray classification (Normal vs Pneumonia)")

    img_file = uploaded_xray or st.file_uploader("Upload X-ray image", type=["png","jpg","jpeg"])
    if img_file:
        try:
            cnn_model = load_keras(MODEL_FILES["cnn"]) if os.path.exists(MODEL_FILES["cnn"]) else None
            img = Image.open(img_file).convert("RGB")
            st.image(img, use_column_width=True)
            if cnn_model is None:
                st.warning("CNN model not found (models/cnn_model_v1.h5).")
            else:
                # preprocess: resize to model expected input
                try:
                    # get expected size from model.input_shape
                    input_shape = cnn_model.input_shape  # e.g. (None, 128, 128, 3)
                    _, h, w, ch = input_shape[0]
                    img_resized = img.resize((w, h))
                except Exception:
                    # fallback default
                    img_resized = img.resize((128, 128))
                arr = np.array(img_resized).astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds = cnn_model.predict(arr)
                # Interpret output:
                label = "Unknown"
                try:
                    if preds.shape[-1] == 1:
                        prob = float(preds[0][0])
                        label = "Pneumonia" if prob > 0.5 else "Normal"
                        st.metric("Prediction", label, f"prob={prob:.2f}")
                    else:
                        # multiclass probabilities
                        idx = int(np.argmax(preds[0]))
                        label = f"Class {idx}"
                        st.metric("Prediction", label, f"prob={preds[0][idx]:.2f}")
                except Exception as e:
                    st.error("Interpretation error: " + str(e))
                    st.text(traceback.format_exc())
        except Exception as e:
            st.error("Failed processing image: " + str(e))
            st.text(traceback.format_exc())

# -----------------------
# Tab: Chatbot / Translator (Gemini)
# -----------------------
with tabs[3]:
    st.header("Chatbot / Translator (Gemini)")
    q = st.text_area("Ask the assistant (short answer first; check explain for details)")
    explain = st.checkbox("Show detailed explanation automatically")
    target_lang = st.text_input("Translator: target language (leave empty to skip)")

    if st.button("Send to Gemini"):
        if genai is None or GENAI_MODEL is None:
            st.error("Gemini not configured. Put GENAI_API_KEY and GENAI_MODEL in Streamlit secrets.")
        else:
            try:
                # short/brief answer first
                prompt_brief = f"{q}\nAnswer concisely in 1-2 short sentences."
                resp_brief = genai.generate(model=GENAI_MODEL, prompt=prompt_brief)
                # new google generative ai uses .text or structured content depending on version
                brief_text = getattr(resp_brief, "text", None) or resp_brief
                st.success("Brief answer:")
                st.write(brief_text)

                if explain:
                    prompt_full = f"{q}\nExplain in detail with examples and citations if applicable."
                    resp_full = genai.generate(model=GENAI_MODEL, prompt=prompt_full)
                    full_text = getattr(resp_full, "text", None) or resp_full
                    st.info("Detailed explanation:")
                    st.write(full_text)

                if target_lang:
                    prompt_trans = f"Translate into {target_lang}: {q}"
                    resp_trans = genai.generate(model=GENAI_MODEL, prompt=prompt_trans)
                    trans_text = getattr(resp_trans, "text", None) or resp_trans
                    st.write("Translation:")
                    st.write(trans_text)
            except Exception as e:
                st.error("Gemini request failed: " + str(e))
                st.text(traceback.format_exc())

# -----------------------
# Tab: Sentiment
# -----------------------
with tabs[4]:
    st.header("Sentiment analysis of feedback")
    feedback = st.text_area("Enter patient feedback / review")
    if st.button("Analyze sentiment"):
        # try local model first
        sent_model = load_joblib(MODEL_FILES["sentiment_model"]) if os.path.exists(MODEL_FILES["sentiment_model"]) else None
        sent_vec = load_joblib(MODEL_FILES["sentiment_vectorizer"]) if os.path.exists(MODEL_FILES["sentiment_vectorizer"]) else None
        if sent_model is not None and sent_vec is not None:
            try:
                X = sent_vec.transform([feedback])
                p = sent_model.predict(X)[0]
                st.success("Positive" if p == 1 else "Negative")
            except Exception as e:
                st.error("Local sentiment failed: " + str(e))
                st.text(traceback.format_exc())
        else:
            # fallback to Gemini classification
            if genai is None or GENAI_MODEL is None:
                st.warning("No sentiment model and Gemini not configured.")
            else:
                try:
                    prompt = f"Classify sentiment (Positive/Negative) for this text: {feedback}. Reply only 'Positive' or 'Negative' with a one-line reason."
                    resp = genai.generate(model=GENAI_MODEL, prompt=prompt)
                    text = getattr(resp, "text", None) or resp
                    st.write(text)
                except Exception as e:
                    st.error("Gemini sentiment failed: " + str(e))
                    st.text(traceback.format_exc())

# -----------------------
# Tab: Clustering / Association (basic UI)
# -----------------------
with tabs[5]:
    st.header("Patient segmentation & association rules (basic)")
    st.write("Upload a tabular CSV (EHR) to run clustering or association mining locally.")
    csf = st.file_uploader("Upload clinical CSV for clustering/association", type=["csv"], key="clust_csv")
    if csf:
        dfc = pd.read_csv(csf)
        st.dataframe(dfc.head())
        if st.button("Run KMeans (k=3)"):
            try:
                from sklearn.cluster import KMeans
                X = dfc.select_dtypes(include=[np.number]).fillna(0).values
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
                st.write("Cluster counts:", np.bincount(kmeans.labels_))
            except Exception as e:
                st.error("Clustering failed: " + str(e))
                st.text(traceback.format_exc())

st.caption("Tip: put trained models in the repository under /models and set secrets: GENAI_API_KEY & GENAI_MODEL.")
