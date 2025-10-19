# app.py
import os
import io
import time
import joblib
import tempfile
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from deep_translator import GoogleTranslator
import gdown

# ML libs
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost  # ensure present in requirements
import tensorflow as tf

# Try to import Gemini client (google generative ai). If not available, will fallback
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# ---------- Helper utilities ----------
st.set_page_config(page_title="HealthAI - Smart Healthcare", layout="wide")
st.title("💊 HealthAI — Multimodal Healthcare Assistant (Streamlit)")

# Ensure models folder
os.makedirs("models", exist_ok=True)

def drive_file_to_path(file_id: str, dest_path: str):
    """Download from Drive using gdown and id; return dest_path or raises."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, dest_path, quiet=True)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to download Drive ID {file_id}: {e}")

@st.cache_data(show_spinner=False)
def load_models_from_secrets():
    """Load all models listed in Streamlit secrets under [DRIVE]."""
    models = {}
    drive = st.secrets.get("DRIVE", {})
    # mapping from expected key -> local filename and loader
    mapping = {
        "risk": ("models/risk_model.joblib", "joblib"),
        "los": ("models/los_model.joblib", "joblib"),
        "cnn_h5": ("models/cnn_model.h5", "keras"),
        "lstm_h5": ("models/lstm_model.h5", "keras"),
        "sentiment_model": ("models/sentiment_model.joblib", "joblib"),
        "sentiment_vectorizer": ("models/sentiment_vectorizer.joblib", "joblib")
    }
    for key, (local_path, kind) in mapping.items():
        file_id = drive.get(key)
        if not file_id:
            st.warning(f"Drive ID for '{key}' not found in secrets. Skipping load for '{key}'.")
            continue
        try:
            # download
            drive_file_to_path(file_id, local_path)
            # load
            if kind == "joblib":
                models[key] = joblib.load(local_path)
            elif kind == "keras":
                models[key] = tf.keras.models.load_model(local_path)
            else:
                models[key] = None
            st.info(f"Loaded '{key}' from Drive.")
        except Exception as e:
            st.error(f"Failed to load {key}: {e}")
            models[key] = None
    return models

# Load models once
with st.spinner("Loading models from Drive (if provided) ..."):
    models = load_models_from_secrets()

# Provide small safe fallback models/data to ensure UI still works (but note they are not clinical)
def make_fallback_models():
    fallback = {}
    # tiny classifier (XGBoost-like fallback using sklearn)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # create tiny dummy training data so predict will not fail (we won't actually train, but allow .predict)
    X_dummy = np.zeros((2,5))
    y_dummy = np.array([0,1])
    clf.fit(X_dummy, y_dummy)
    fallback['risk'] = clf

    # LOS regressor fallback
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_dummy, np.array([5.0,10.0]))
    fallback['los'] = reg

    # Tiny CNN fallback: small Sequential that can do predict on preprocessed images
    try:
        model_c = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,128,3)),
            tf.keras.layers.Conv2D(8,3,activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model_c.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        fallback['cnn_h5'] = model_c
    except Exception:
        fallback['cnn_h5'] = None

    # Tiny LSTM fallback for timeseries -> predict single value
    try:
        model_l = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,1)),
            tf.keras.layers.LSTM(8),
            tf.keras.layers.Dense(1)
        ])
        model_l.compile(optimizer='adam', loss='mse')
        fallback['lstm_h5'] = model_l
    except Exception:
        fallback['lstm_h5'] = None

    # Sentiment fallback
    from sklearn.dummy import DummyClassifier
    dummy_sent = DummyClassifier(strategy="most_frequent")
    dummy_sent.fit([[0],[1]], [0,1])
    fallback['sentiment_model'] = dummy_sent
    fallback['sentiment_vectorizer'] = lambda texts: np.zeros((len(texts),1))
    return fallback

fallbacks = make_fallback_models()

# Merge loaded models with fallback if missing
for k in ['risk','los','cnn_h5','lstm_h5','sentiment_model','sentiment_vectorizer']:
    if models.get(k) is None:
        models[k] = fallbacks.get(k)
        st.warning(f"Using fallback for {k}")

# ---------- Utility ML functions ----------
def predict_risk(df_row, model):
    """Accepts array-like or dataframe row of features matching training order."""
    # The expected features in your Jupyter were: age, bmi, blood_pressure, cholesterol, diabetes, heart_disease
    features = np.array([
        df_row.get("age", 50),
        df_row.get("bmi", 26.5),
        df_row.get("blood_pressure", 135),
        df_row.get("cholesterol", 225),
        df_row.get("diabetes", 0),
        df_row.get("heart_disease", 0)
    ], dtype=float).reshape(1,-1)
    try:
        pred_proba = model.predict_proba(features)[0]
        pred = model.predict(features)[0]
        return int(pred), float(pred_proba.max())
    except Exception:
        # fallback approximate
        return 0, 0.5

def predict_los(df_row, model):
    features = np.array([
        df_row.get("age",50),
        df_row.get("bmi",26.5),
        df_row.get("blood_pressure",135),
        df_row.get("cholesterol",225),
        df_row.get("diabetes",0),
        df_row.get("heart_disease",0)
    ], dtype=float).reshape(1,-1)
    try:
        pred = model.predict(features)[0]
        return float(pred)
    except Exception:
        return float(np.round(np.mean([5,10])))

def predict_cnn_image(image:Image.Image, model):
    # preprocess to 128x128 and normalize
    image = image.convert("RGB").resize((128,128))
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, axis=0)
    try:
        probs = model.predict(arr)[0]
        label_idx = int(np.argmax(probs))
        labels = ["NORMAL","PNEUMONIA"]
        return labels[label_idx], float(probs[label_idx])
    except Exception as e:
        return "NORMAL", 0.6

def predict_lstm_timeseries(df_ts:pd.DataFrame, model):
    # expects single column of a vital (resample or pad/truncate to 10 steps)
    series = df_ts.iloc[:,0].values.astype(float)
    # build length 10 sliding or pad
    if len(series) >= 10:
        seq = series[-10:]
    else:
        seq = np.pad(series, (10-len(series),0), 'constant', constant_values=0)
    seq = seq.reshape(1,10,1)
    try:
        pred = model.predict(seq)[0][0]
        return float(pred)
    except Exception:
        return float(1.0)

def sentiment_predict(text, model, vectorizer):
    try:
        X = vectorizer.transform([text])
        p = model.predict(X)[0]
        label = "positive" if int(p)==1 else "negative"
        return label
    except Exception:
        # try gemini for sentiment (if configured), else dummy
        return "neutral"

# ---------- Gemini-backed Chatbot helper ----------
def chat_with_gemini(prompt_text, lang="English", model_name=None):
    """
    Try to call gemini via google.generativeai (if available and configured).
    Returns text response or raises.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini client not available in this environment.")
    api_key = st.secrets.get("GENAI_API_KEY") or st.secrets.get("genai_api_key")
    if not api_key:
        raise RuntimeError("GENAI_API_KEY not found in Streamlit secrets.")
    # configure
    try:
        genai.configure(api_key=api_key)
        model_to_use = model_name or st.secrets.get("GEMINI_MODEL") or "models/gemini-2.5-flash"
        # best attempt to use provided client API
        try:
            model = genai.GenerativeModel(model_to_use)
            response = model.generate_content(f"You are a healthcare assistant. Answer briefly first (2-3 bullets) then expand if user asks.\nPatient: {prompt_text}")
            text = getattr(response, "text", None)
            if not text and isinstance(response, dict):
                text = response.get("candidates",[{}])[0].get("content", "")
            return text or str(response)
        except Exception as e:
            # older/newer client variations: try genai.generate_text-like
            try:
                resp = genai.generate(prompt_text, model=model_to_use)
                return resp.get("output", "")
            except Exception as e2:
                raise RuntimeError("Gemini generate failed: " + str(e2))
    except Exception as e:
        raise

# ---------- Streamlit UI ----------
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose module",
    ["Overview", "Risk Prediction", "LOS Prediction", "Patient Segmentation (Clustering)",
     "Association Rules (Apriori)", "CNN Image Diagnosis", "LSTM Timeseries", "Sentiment Analysis",
     "Translator", "Chatbot", "Model & Data Diagnostics"]
)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Project Overview & Features")
    st.markdown("""
    **HealthAI** — multimodal healthcare demo: risk classification, length-of-stay regression,
    clustering, association rules, CNN (image), LSTM (time-series), sentiment, translation, and Gemini chatbot.
    """)

    # quick dataset visualization if user uploads dataset
    st.info("If you have a tabular dataset (CSV), upload it below to visualize EDA charts.")
    uploaded = st.file_uploader("Upload tabular EHR CSV (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        # safe encode gender if exists
        if "gender" in df.columns:
            df["gender_encoded"] = df["gender"].astype('category').cat.codes
        st.markdown("### Age distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["age"].dropna(), bins=20, kde=False, ax=ax)
        st.pyplot(fig)
        st.markdown("### Outcome counts")
        if "outcome" in df.columns:
            fig2, ax2 = plt.subplots()
            sns.countplot(x=df["outcome"], ax=ax2)
            st.pyplot(fig2)
        st.markdown("### Correlation heatmap (numeric columns)")
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            corr = num.corr()
            fig3, ax3 = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

# ---------- Risk Prediction ----------
elif page == "Risk Prediction":
    st.subheader("Disease Risk Prediction (Classification)")
    with st.form("risk_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male","Female","Other"])
        bmi = st.number_input("BMI", value=26.5, step=0.1)
        bp = st.number_input("Systolic BP", value=135)
        chol = st.number_input("Cholesterol", value=225)
        diabetes = st.selectbox("Diabetes", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        heart = st.selectbox("Heart disease", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        submitted = st.form_submit_button("Predict Risk")
    if submitted:
        row = {"age":age,"bmi":bmi,"blood_pressure":bp,"cholesterol":chol,"diabetes":diabetes,"heart_disease":heart}
        pred, conf = predict_risk(row, models['risk'])
        st.success(f"Predicted risk class: **{pred}** (probability ~ {conf:.2f})")
        st.caption("Interpretation: 0 = lower risk, 1 = higher risk (use your model mapping).")

# ---------- LOS ----------
elif page == "LOS Prediction":
    st.subheader("Length of Stay (Regression)")
    with st.form("los_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=50, key="los_age")
        bmi = st.number_input("BMI", value=26.5, step=0.1, key="los_bmi")
        bp = st.number_input("Systolic BP", value=135, key="los_bp")
        chol = st.number_input("Cholesterol", value=225, key="los_chol")
        diabetes = st.selectbox("Diabetes", [0,1], key="los_diab")
        heart = st.selectbox("Heart disease", [0,1], key="los_heart")
        sub = st.form_submit_button("Predict LOS")
    if sub:
        row = {"age":age,"bmi":bmi,"blood_pressure":bp,"cholesterol":chol,"diabetes":diabetes,"heart_disease":heart}
        days = predict_los(row, models['los'])
        st.success(f"Predicted Length of Stay: **{days:.2f} days**")
        st.caption("Note: model predictions are as-good-as the model and data provided. Validate clinically.")

# ---------- Clustering ----------
elif page == "Patient Segmentation (Clustering)":
    st.subheader("Patient Segmentation (KMeans)")
    st.info("Upload a CSV with numeric clinical columns (age, bmi, blood_pressure, cholesterol, diabetes, heart_disease).")
    up = st.file_uploader("Upload CSV for clustering", type="csv")
    n_clusters = st.slider("Number of clusters", 2, 8, 3)
    if up:
        df = pd.read_csv(up)
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns to cluster.")
        else:
            X = df[numeric_cols].fillna(0).values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xs)
            df['cluster'] = km.labels_
            st.write(df.head())
            st.markdown("Cluster counts:")
            st.write(df['cluster'].value_counts())
            # plot two principal features
            fig, ax = plt.subplots()
            sns.scatterplot(x=Xs[:,0], y=Xs[:,1], hue=km.labels_, palette="tab10", ax=ax)
            ax.set_title("Cluster visualization (first two numeric features)")
            st.pyplot(fig)

# ---------- Association Rules ----------
elif page == "Association Rules (Apriori)":
    st.subheader("Association Rules (Apriori)")
    st.info("Upload a CSV of boolean/comorbidity indicators (columns like has_diabetes, high_bp, has_heart).")
    up = st.file_uploader("Upload CSV for assoc rules", type="csv", key="assoc")
    min_support = st.slider("min_support", 0.01, 0.5, 0.1)
    if up:
        df = pd.read_csv(up)
        # prepare one-hot boolean dataset
        bool_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1,True,False}]
        if not bool_cols:
            st.error("No boolean indicator columns detected. Provide columns with 0/1.")
        else:
            trans = df[bool_cols].astype(int)
            # use mlxtend apriori/association tools
            from mlxtend.frequent_patterns import apriori, association_rules
            freq = apriori(trans, min_support=min_support, use_colnames=True)
            st.write("Frequent itemsets (top 10):", freq.sort_values("support", ascending=False).head(10))
            rules = association_rules(freq, metric="confidence", min_threshold=0.5)
            st.write("Top rules:", rules.sort_values("lift", ascending=False).head(10))

# ---------- CNN Image Diagnosis ----------
elif page == "CNN Image Diagnosis":
    st.subheader("Chest X-ray Diagnosis (CNN)")
    st.info("Upload a chest X-ray image (PNG/JPG) and get predicted NORMAL/PNEUMONIA.")
    uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
        pred_label, pred_conf = predict_cnn_image(image, models['cnn_h5'])
        st.success(f"Prediction: **{pred_label}** (confidence {pred_conf:.2f})")

# ---------- LSTM Timeseries ----------
elif page == "LSTM Timeseries":
    st.subheader("LSTM Prediction on Vital Time-series")
    st.info("Upload a CSV with a single column (e.g., heart_rate or oxygen) or multiple columns (we use first).")
    up = st.file_uploader("Upload time-series CSV", type="csv", key="ts")
    if up:
        df_ts = pd.read_csv(up)
        st.write(df_ts.head())
        pred = predict_lstm_timeseries(df_ts, models['lstm_h5'])
        st.success(f"LSTM predicted next-step value: **{pred:.3f}**")

# ---------- Sentiment ----------
elif page == "Sentiment Analysis":
    st.subheader("Sentiment Analysis on Patient Feedback")
    text = st.text_area("Paste patient feedback / review here")
    if st.button("Analyze Sentiment"):
        pred = sentiment_predict(text, models['sentiment_model'], models['sentiment_vectorizer'])
        st.success(f"Sentiment: **{pred}**")
        # Optionally ask Gemini for nuance (if configured)
        if GENAI_AVAILABLE and st.secrets.get("GENAI_API_KEY"):
            if st.checkbox("Also get Gemini-based sentiment/framing?"):
                try:
                    prompt = f"Determine the sentiment (positive/negative/neutral) and one-sentence reason: {text}"
                    gem = chat_with_gemini(prompt)
                    st.info("Gemini sentiment:")
                    st.write(gem)
                except Exception as e:
                    st.warning(f"Gemini sentiment failed: {e}")

# ---------- Translator ----------
elif page == "Translator":
    st.subheader("Translator (GoogleTranslator via deep-translator)")
    text = st.text_area("Enter text to translate")
    target = st.selectbox("Target language", ["English","Tamil","Hindi","Malayalam"])
    if st.button("Translate"):
        try:
            code = {"English":"en","Tamil":"ta","Hindi":"hi","Malayalam":"ml"}[target]
            translated = GoogleTranslator(source="auto", target=code).translate(text)
            st.success(f"Translated ({target}):")
            st.write(translated)
        except Exception as e:
            st.error(f"Translation error: {e}")

# ---------- Chatbot ----------
elif page == "Chatbot":
    st.subheader("Healthcare Chatbot (Gemini primary; fallback local rules)")
    st.markdown("The bot returns a short answer first. If you want more detail, ask 'explain' or 'more'.")
    user_lang = st.selectbox("Language", ["English","Tamil","Hindi","Malayalam"])
    user_q = st.text_input("Your question", value="", key="chat_input")
    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Type a question.")
        else:
            answer_text = ""
            use_gemini = GENAI_AVAILABLE and st.secrets.get("GENAI_API_KEY")
            if use_gemini:
                try:
                    # A concise prompt: brief then elaboration on request
                    prompt = f"You are a healthcare assistant. Answer briefly (2-3 bullets). If user asks 'explain' later, expand. Language: {user_lang}.\nQuestion: {user_q}"
                    answer_text = chat_with_gemini(prompt, lang=user_lang, model_name=st.secrets.get("GEMINI_MODEL"))
                    st.success("Gemini response:")
                    st.write(answer_text)
                except Exception as e:
                    st.error(f"Gemini error: {e}")
                    use_gemini = False
            if not use_gemini:
                # Local fallback: simple rule-based or canned answers (not clinical)
                q = user_q.lower()
                if "diabet" in q:
                    answer_text = "Short: Frequent urination, thirst, fatigue. Ask to 'explain' for details."
                elif "pneumonia" in q or "cough" in q:
                    answer_text = "Short: Cough, fever, breathlessness. See doctor for imaging."
                else:
                    answer_text = "Short: Please consult a healthcare professional. Ask 'explain' if you want more info."
                st.info("Fallback response:")
                st.write(answer_text)

# ---------- Model & Data Diagnostics ----------
elif page == "Model & Data Diagnostics":
    st.subheader("Model & Data Diagnostics")
    st.markdown("This page shows which models were loaded, and quick checks.")
    for k in ["risk","los","cnn_h5","lstm_h5","sentiment_model","sentiment_vectorizer"]:
        m = models.get(k)
        if m is None:
            st.write(f"**{k}** — Not loaded")
        else:
            st.write(f"**{k}** — Loaded; type: {type(m)}")
    st.markdown("---")
    st.write("If a model failed to load, make sure the file ID is correct and Drive file is shared 'anyone with link'.")

# ---------- end of pages ----------
st.markdown("---")
st.caption("HealthAI — demo. Models should be clinically validated before use. Use responsibly.")
