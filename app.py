import streamlit as st
import pandas as pd
import joblib

# --- Custom Styling for School Colors (Dark Gray & Red Accents) ---
st.markdown(
    """
    <style>
    .block-container { background-color: #1f1f1f; color: #f0f0f0; padding: 2rem; border-radius: 0.5rem; }
    h1 { color: #e3120b; font-size: 2.5rem; border-bottom: 3px solid #e3120b; padding-bottom: 0.5rem; }
    h2, h3, p { color: #e0e0e0; }
    .stFileUploader>div>div>div { background-color: #2b2b2b; border: 2px solid #2b2b2b; transition: border-color 0.3s ease; }
    .stFileUploader>div>div>div:hover { border-color: #e3120b; }
    .stButton>button { background-color: #e3120b !important; color: #fff !important; border-radius: 0.25rem; padding: 0.5rem 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
    .css-1d391kg .css-1v3fvcr { background-color: #2b2b2b; border-radius: 0.5rem; padding: 1rem; }
    .css-1d391kg h2 { color: #e3120b; }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.title("Rocket Engine Remaining Useful Life Predictor")
st.write("Upload up to 5 CSV or TXT files with engine sensor readings to get RUL predictions.")

# Create 5 upload slots
cols = st.columns(5)
uploaders = []
for i, col in enumerate(cols, start=1):
    with col:
        file = st.file_uploader(f"Upload file {i}", type=["csv","txt"], key=f"file{i}")
        uploaders.append(file)

# Load model
model = None
try:
    model = joblib.load("model.pkl")
except:
    st.warning("Model file not found. Please add 'model.pkl' to the repo.")

# Helper to read space-delimited TXT or comma CSV
def read_sensor_file(fp):
    fname = fp.name.lower()
    if fname.endswith(".txt"):
        return pd.read_csv(fp, sep=r"\s+", header=None)
    else:
        return pd.read_csv(fp)

# Process each upload
for idx, uploaded in enumerate(uploaders, start=1):
    if uploaded:
        df = read_sensor_file(uploaded)
        st.subheader(f"Preview of file {idx}")
        st.dataframe(df.head())

        if model:
            preds = model.predict(df)
            df["Predicted RUL"] = preds
            st.subheader(f"Predictions for file {idx}")
            st.line_chart(df["Predicted RUL"])
        else:
            st.error("Cannot predict without a loaded model.")

# Sidebar
st.sidebar.header("Repository Setup & Deployment")
st.sidebar.markdown(
    "1. Fork this repo on GitHub.\n"
    "2. Add your trained model as `model.pkl`.\n"
    "3. Ensure `requirements.txt` lists all dependencies.\n"
    "4. Deploy via Streamlit Cloud or your hosting of choice."
)
