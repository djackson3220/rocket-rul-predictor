import streamlit as st
import pandas as pd
import joblib

# --- Styling (Dark Gray & Red Accents) ---
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

# --- App Header ---
st.title("Rocket Engine Remaining Useful Life Predictor")
st.write("Upload up to 5 CSV or TXT files with engine sensor readings to get RUL predictions.")

# --- File Uploaders ---
cols = st.columns(5)
uploaders = []
for i, col in enumerate(cols, start=1):
    with col:
        file = st.file_uploader(f"Upload file {i}", type=["csv","txt"], key=f"file{i}")
        uploaders.append(file)

# --- Load the Model ---
model = None
try:
    model = joblib.load("model.pkl")
except:
    st.warning("Model file not found. Please add 'model.pkl' to the repo.")

# --- Column Names for C-MAPSS ---
cmapss_cols = (
    ["engine_id", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# --- Reader Function ---
def read_sensor_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        df = pd.read_csv(uploaded, sep=r"\s+", header=None)
        df.columns = cmapss_cols
        return df
    else:
        df = pd.read_csv(uploaded)
        # If the CSV already has headers, trust them; otherwise you could assign cmapss_cols here too.
        return df

# --- Process Each Upload ---
for idx, uploaded in enumerate(uploaders, start=1):
    if uploaded is not None:
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

# --- Sidebar Instructions ---
st.sidebar.header("Repository Setup & Deployment")
st.sidebar.markdown(
    "1. Fork this repo on GitHub.\n"
    "2. Add your trained model as `model.pkl`.\n"
    "3. Ensure `requirements.txt` lists all dependencies.\n"
    "4. Deploy via Streamlit Cloud or your hosting of choice."
)
