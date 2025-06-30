import streamlit as st
import pandas as pd
import joblib

# --- Custom Styling for Black & Red Theme ---
st.markdown(
    """
    <style>
    /* Page background */
    .block-container { background-color: #000000; color: #ffffff; }
    /* Header styling */
    h1 { color: #e3120b; font-size: 2.5rem; margin-bottom: 0.5rem; }
    /* Subheader and text */
    h2, h3, p { color: #ffffff; }
    /* File uploader */
    .stFileUploader>div>div>div { background-color: #1a1a1a; border: 1px solid #e3120b; }
    /* Button styling */
    .stButton>button { background-color: #e3120b; color: #ffffff; border-radius: 0.25rem; padding: 0.5rem 1rem; }
    /* Sidebar background */
    .css-1d391kg .css-1v3fvcr { background-color: #1a1a1a; }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description (with custom header)
st.title("Rocket Engine Remaining Useful Life Predictor")
st.write(
    "Upload your sensor data file and get a prediction of the engine's Remaining Useful Life (RUL)."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file with engine sensor readings",
    type=["csv"]
)

# Load pre-trained model (replace 'model.pkl' with actual path)
model = None
try:
    model = joblib.load("model.pkl")
except Exception:
    st.warning("Model file not found. Please add 'model.pkl' to the repo.")

if uploaded_file:
    # Read and display data
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if model:
        predictions = model.predict(df)
        df['Predicted RUL'] = predictions
        st.subheader("Predicted Remaining Useful Life")
        st.line_chart(df['Predicted RUL'])
    else:
        st.error("Cannot predict without a loaded model.")

# Sidebar instructions
st.sidebar.header("Repository Setup")
st.sidebar.markdown(
    "1. Fork this repo on GitHub.\n"
    "2. Add your trained model as `model.pkl` in the project root.\n"
    "3. Update `requirements.txt` with dependencies (`streamlit`, `pandas`, `scikit-learn`, `joblib`).\n"
    "4. Deploy on Streamlit Cloud or your preferred hosting."
)
